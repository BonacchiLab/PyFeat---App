"""Stimulus-level summaries and event-aligned signal analytics."""
from typing import Any, Mapping

import pandas as pd


def visualization_time(df: pd.DataFrame):
    for column in ("time_sec", "time_start_sec", "frame"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce"), column
    return pd.Series(range(len(df)), index=df.index, dtype="float64"), "sample"


def compare_stimulus_windows(results: Mapping[str, Any], metric: str) -> pd.DataFrame:
    records = []
    for result in results.values():
        df = result["dataframe"]
        if metric not in df.columns:
            continue
        times, _ = visualization_time(df)
        values = pd.to_numeric(df[metric], errors="coerce")
        for stimulus in result.get("stimuli", []):
            start, end = float(stimulus["inicio"]), float(stimulus["fim"])
            if end < start:
                start, end = end, start
            before = values.loc[times.ge(start - 5) & times.lt(start)].mean()
            during = values.loc[times.ge(start) & times.lt(end)].mean()
            if pd.isna(before) or pd.isna(during):
                continue
            case = f"{result['subject_id']} · {df['input_file'].iloc[0]} · {stimulus['nome']}"
            for period, value in (("Before", before), ("During", during)):
                records.append({"case": case, "subject": result["subject_id"], "stimulus": stimulus["nome"], "period": period, "value": value})
    return pd.DataFrame.from_records(records)


def event_aligned_signals(results: Mapping[str, Any], metric: str, stimulus_name: str, padding: float = 5.0) -> pd.DataFrame:
    records = []
    for result in results.values():
        df = result["dataframe"]
        if metric not in df.columns or df.empty:
            continue
        stimulus = next((item for item in result.get("stimuli", []) if item["nome"] == stimulus_name), None)
        if stimulus is None:
            continue
        start, end = float(stimulus["inicio"]), float(stimulus["fim"])
        if end < start:
            start, end = end, start
        times, time_column = visualization_time(df)
        values = pd.to_numeric(df[metric], errors="coerce")
        window = times.ge(start - padding) & times.le(end + padding)
        case = f"{result['subject_id']} · {df['input_file'].iloc[0]}"
        for relative_time, value in zip(times.loc[window] - start, values.loc[window]):
            if pd.isna(relative_time) or pd.isna(value):
                continue
            records.append({"relative_time": float(relative_time), "value": float(value), "recording": case, "time_column": time_column, "stimulus_duration": end - start})
    return pd.DataFrame.from_records(records)
