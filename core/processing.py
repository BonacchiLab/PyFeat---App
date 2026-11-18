"""Data preparation and export functions used by the FEDET API."""

from io import BytesIO
from typing import Any, Iterable, Mapping

import pandas as pd


def format_seconds_to_clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_skip_frames(video_fps: float, target_fps: float) -> int:
    return max(1, int(round(float(video_fps) / float(target_fps))))


def aggregate_by_second(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Aggregate per-frame rows to seconds using the mean of numeric columns."""
    if df is None or df.empty or "frame" not in df.columns:
        return df

    df = df.copy()
    df["_frame_index"] = pd.to_numeric(df["frame"], errors="coerce")
    df["_second_bin"] = (df["_frame_index"] / float(fps)).astype(float).floordiv(1).astype(int)
    numeric_cols = [
        col for col in df.columns
        if col not in {"input", "frame", "approx_time", "_frame_index", "_second_bin"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        return df

    grouped = df.groupby("_second_bin", as_index=False)[numeric_cols].mean()
    grouped["second"] = grouped["_second_bin"].astype(int)
    grouped["time_start_sec"] = grouped["second"].astype(float)
    grouped["time_end_sec"] = grouped["second"].astype(float) + 1.0
    grouped["approx_time"] = grouped["time_start_sec"].apply(format_seconds_to_clock)
    grouped["frame"] = grouped["time_start_sec"] * float(fps)
    if "input" in df.columns:
        grouped["input"] = df.groupby("_second_bin")["input"].first().reset_index(drop=True).to_numpy()
    return grouped.sort_values("second").reset_index(drop=True)


def prepare_frame_dataframe(df_raw: pd.DataFrame, fps: float, sampling_mode: str) -> pd.DataFrame:
    """Return per-frame rows or aggregate the 1 FPS representative samples."""
    if sampling_mode in {"all_frames", "custom_fps"}:
        df_out = df_raw.copy()
        df_out["time_sec"] = pd.to_numeric(df_out["frame"], errors="coerce") / float(fps)
        df_out["approx_time"] = df_out["time_sec"].apply(format_seconds_to_clock)
        return df_out
    return aggregate_by_second(df_raw, fps)


def apply_stimulus_metadata(df: pd.DataFrame, stimuli: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Label rows falling inside configured stimulus windows."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "time_sec" in df.columns:
        row_times = pd.to_numeric(df["time_sec"], errors="coerce")
    elif "time_start_sec" in df.columns:
        row_times = pd.to_numeric(df["time_start_sec"], errors="coerce")
    elif "frame" in df.columns:
        row_times = pd.to_numeric(df["frame"], errors="coerce")
    else:
        row_times = pd.Series(float("nan"), index=df.index)
    for index, stimulus in enumerate(stimuli, start=1):
        start = float(stimulus["inicio"])
        end = float(stimulus["fim"])
        if end < start:
            start, end = end, start
        matches = row_times.ge(start) & row_times.lt(end)
        labels = pd.Series("", index=df.index, dtype="object")
        labels.loc[matches] = stimulus["nome"]
        df[f"Stimulus_{index}"] = labels
    return df


def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    return buffer.getvalue()
