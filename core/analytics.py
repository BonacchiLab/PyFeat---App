"""Stimulus-level summaries and trajectory similarity analytics."""

import math
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


def pearson_correlation(left_values, right_values):
    if len(left_values) < 4:
        return None
    left_mean, right_mean = sum(left_values) / len(left_values), sum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_sum, right_sum = sum(value * value for value in left_centered), sum(value * value for value in right_centered)
    if left_sum == 0 or right_sum == 0:
        return None
    covariance = sum(a * b for a, b in zip(left_centered, right_centered))
    return covariance / math.sqrt(left_sum * right_sum)


def stimulus_similarity_matrix(results: Mapping[str, Any], metric: str, stimulus_name: str, phase_bins: int = 20):
    recording_series = {}
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
        if end <= start:
            continue
        times, _ = visualization_time(df)
        values = pd.to_numeric(df[metric], errors="coerce")
        within = times.ge(start) & times.lt(end) & values.notna()
        phase_data = pd.DataFrame({"phase": (times.loc[within] - start) / (end - start), "value": values.loc[within]})
        if phase_data.empty:
            continue
        phase_data["phase_bin"] = (phase_data["phase"] * phase_bins).astype(int).clip(0, phase_bins - 1)
        binned = phase_data.groupby("phase_bin")["value"].mean().to_dict()
        if len(binned) < 4:
            continue
        recording_series[f"{result['subject_id']} · {df['input_file'].iloc[0]}"] = binned
    labels = list(recording_series)
    correlations, shared_bin_counts = [], []
    for left in labels:
        corr_row, count_row = [], []
        for right in labels:
            common = sorted(set(recording_series[left]) & set(recording_series[right]))
            corr_row.append(pearson_correlation([recording_series[left][i] for i in common], [recording_series[right][i] for i in common]))
            count_row.append(len(common))
        correlations.append(corr_row)
        shared_bin_counts.append(count_row)
    return (pd.DataFrame(correlations, index=labels, columns=labels, dtype="float64"),
            pd.DataFrame(shared_bin_counts, index=labels, columns=labels, dtype="int64"))
