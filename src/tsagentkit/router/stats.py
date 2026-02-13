"""Routing statistics extraction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsagentkit.contracts import RouterThresholds, TaskSpec
from tsagentkit.time import normalize_pandas_freq


def compute_router_stats(
    df: pd.DataFrame,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
) -> tuple[dict[str, float], list[str]]:
    """Compute routing statistics and buckets for panel data."""
    return _compute_series_stats_internal(df, task_spec, thresholds, is_single_series=False)


def compute_series_stats(
    series_df: pd.DataFrame,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
) -> tuple[dict[str, float], list[str]]:
    """Compute routing statistics and buckets for a single series."""
    return _compute_series_stats_internal(series_df, task_spec, thresholds, is_single_series=True)


def _compute_series_stats_internal(
    df: pd.DataFrame,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
    *,
    is_single_series: bool,
) -> tuple[dict[str, float], list[str]]:
    stats: dict[str, float] = {}
    buckets: list[str] = []

    if is_single_series:
        length = len(df)
        stats["series_length"] = float(length)
    else:
        lengths = df.groupby("unique_id").size()
        length = int(lengths.min()) if not lengths.empty else 0
        stats["min_series_length"] = float(length)

    if length < thresholds.min_train_size:
        buckets.append("short_history")

    missing_ratio = _compute_missing_ratio(df, task_spec)
    stats["missing_ratio"] = float(missing_ratio)
    if missing_ratio > thresholds.max_missing_ratio:
        buckets.append("sparse")

    uid_col = task_spec.panel_contract.unique_id_col
    ds_col = task_spec.panel_contract.ds_col
    y_col = task_spec.panel_contract.y_col

    intermittency = _compute_intermittency(df, thresholds, uid_col, ds_col, y_col)
    stats.update(intermittency)
    if intermittency.get("intermittent_series_ratio", 0.0) > 0:
        buckets.append("intermittent")

    season_conf = _seasonality_confidence(df, task_spec, uid_col, y_col)
    stats["seasonality_confidence"] = float(season_conf)
    if season_conf >= thresholds.min_seasonality_conf:
        buckets.append("seasonal_candidate")

    if is_single_series:
        trend_strength = _compute_trend_strength(df, y_col)
        stats["trend_strength"] = float(trend_strength)
        if trend_strength >= thresholds.min_trend_strength:
            buckets.append("trend")
    else:
        trend_ratios = []
        sample_uids = list(df[uid_col].unique())[:10]
        for uid in sample_uids:
            series = df[df[uid_col] == uid].sort_values(ds_col)
            trend_strength = _compute_trend_strength(series, y_col)
            trend_ratios.append(trend_strength)
        avg_trend = float(np.mean(trend_ratios)) if trend_ratios else 0.0
        stats["trend_strength"] = avg_trend
        if avg_trend >= thresholds.min_trend_strength:
            buckets.append("trend")

    _add_high_frequency_bucket(task_spec, buckets)
    return stats, buckets


def _add_high_frequency_bucket(task_spec: TaskSpec, buckets: list[str]) -> None:
    if task_spec.freq and task_spec.freq.upper() in ("H", "BH", "T", "MIN", "S"):
        buckets.append("high_frequency")


def _compute_trend_strength(df: pd.DataFrame, y_col: str) -> float:
    if df.empty:
        return 0.0

    y = df[y_col].dropna().values
    if len(y) < 3:
        return 0.0

    try:
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    except (ValueError, np.linalg.LinAlgError):
        return 0.0


def _compute_missing_ratio(df: pd.DataFrame, task_spec: TaskSpec) -> float:
    if df.empty:
        return 0.0
    uid_col = task_spec.panel_contract.unique_id_col
    ds_col = task_spec.panel_contract.ds_col

    ratios = []
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid].sort_values(ds_col)
        if series.empty:
            continue
        full_range = pd.date_range(
            start=series[ds_col].min(),
            end=series[ds_col].max(),
            freq=normalize_pandas_freq(task_spec.freq),
        )
        missing = len(full_range) - len(series)
        ratio = missing / max(len(full_range), 1)
        ratios.append(ratio)
    return float(np.mean(ratios)) if ratios else 0.0


def _compute_intermittency(
    df: pd.DataFrame,
    thresholds: RouterThresholds,
    uid_col: str,
    ds_col: str,
    y_col: str,
) -> dict[str, float]:
    intermittent = 0
    total = 0

    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid].sort_values(ds_col)
        y = series[y_col].values
        total += 1

        non_zero_idx = np.where(y > 0)[0]
        if len(non_zero_idx) <= 1:
            adi = float("inf")
            cv2 = float("inf")
        else:
            intervals = np.diff(non_zero_idx)
            adi = float(np.mean(intervals)) if len(intervals) > 0 else float("inf")
            non_zero_vals = y[non_zero_idx]
            mean = np.mean(non_zero_vals) if len(non_zero_vals) > 0 else 0.0
            std = np.std(non_zero_vals) if len(non_zero_vals) > 0 else 0.0
            cv2 = float((std / mean) ** 2) if mean != 0 else float("inf")

        if adi >= thresholds.max_intermittency_adi and cv2 >= thresholds.max_intermittency_cv2:
            intermittent += 1

    ratio = intermittent / total if total > 0 else 0.0
    return {
        "intermittent_series_ratio": ratio,
        "intermittent_series_count": float(intermittent),
    }


def _seasonality_confidence(
    df: pd.DataFrame,
    task_spec: TaskSpec,
    uid_col: str,
    y_col: str,
) -> float:
    season_length = task_spec.season_length
    if season_length is None or season_length <= 1:
        return 0.0
    confs: list[float] = []
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid][y_col].values
        if len(series) <= season_length:
            continue
        series = series - np.mean(series)
        denom = np.dot(series, series)
        if denom == 0:
            continue
        lagged = np.roll(series, season_length)
        corr = np.dot(series[season_length:], lagged[season_length:]) / denom
        confs.append(abs(float(corr)))
    return float(np.mean(confs)) if confs else 0.0


__all__ = ["compute_router_stats", "compute_series_stats"]

