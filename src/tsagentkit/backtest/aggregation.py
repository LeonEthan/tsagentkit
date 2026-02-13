"""Backtest metric aggregation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .report import SegmentMetrics, SeriesMetrics, TemporalMetrics


def summary_to_metrics(summary_df: pd.DataFrame, model_name: str | None) -> dict[str, float]:
    """Convert summary frame into metric mapping for a specific model."""
    if summary_df is None or summary_df.empty:
        return {}
    df = summary_df
    if "model" in df.columns:
        if model_name and model_name in df["model"].unique():
            df = df[df["model"] == model_name]
        else:
            df = df[df["model"] == df["model"].iloc[0]]
    if "metric" not in df.columns or "value" not in df.columns:
        return {}
    metrics = df.groupby("metric")["value"].mean()
    return {metric: float(value) for metric, value in metrics.items()}


def series_metrics_from_frame(
    metrics_df: pd.DataFrame,
    model_name: str | None,
) -> dict[str, dict[str, float]]:
    """Build per-series metrics dictionary from metric frame."""
    if metrics_df is None or metrics_df.empty:
        return {}
    df = metrics_df
    if "model" in df.columns:
        if model_name and model_name in df["model"].unique():
            df = df[df["model"] == model_name]
        else:
            df = df[df["model"] == df["model"].iloc[0]]
    if "unique_id" not in df.columns or "metric" not in df.columns or "value" not in df.columns:
        return {}
    grouped = df.groupby(["unique_id", "metric"])["value"].mean().reset_index()
    result: dict[str, dict[str, float]] = {}
    for uid, group in grouped.groupby("unique_id"):
        result[uid] = {row["metric"]: float(row["value"]) for _, row in group.iterrows()}
    return result


def aggregate_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]]
) -> dict[str, float]:
    """Aggregate metrics across all series and windows."""
    if not series_metrics_agg:
        return {}

    all_metric_names: set[str] = set()
    for metrics_list in series_metrics_agg.values():
        for metrics in metrics_list:
            all_metric_names.update(metrics.keys())

    aggregated: dict[str, float] = {}
    for metric_name in all_metric_names:
        values = []
        for metrics_list in series_metrics_agg.values():
            for metric in metrics_list:
                if metric_name in metric and not np.isnan(metric[metric_name]):
                    values.append(metric[metric_name])

        if values:
            aggregated[metric_name] = float(np.mean(values))
        else:
            aggregated[metric_name] = float("nan")

    return aggregated


def build_series_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]]
) -> dict[str, SeriesMetrics]:
    """Build SeriesMetrics objects from aggregated window metrics."""
    series_metrics: dict[str, SeriesMetrics] = {}
    for uid, metrics_list in series_metrics_agg.items():
        metric_names: set[str] = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        avg_metrics = {
            metric_name: np.mean(
                [window_metrics[metric_name] for window_metrics in metrics_list if not np.isnan(window_metrics.get(metric_name, np.nan))]
            )
            for metric_name in metric_names
        }
        series_metrics[uid] = SeriesMetrics(
            series_id=uid,
            metrics=avg_metrics,
            num_windows=len(metrics_list),
        )
    return series_metrics


def compute_segment_metrics(
    series_metrics: dict[str, SeriesMetrics],
    dataset: object,
) -> dict[str, SegmentMetrics]:
    """Compute segment metrics grouped by sparsity class."""
    from collections import defaultdict

    segment_series: dict[str, list[str]] = defaultdict(list)
    segment_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)

    if dataset.sparsity_profile:
        for uid in series_metrics:
            classification = dataset.sparsity_profile.get_classification(uid)
            segment_name = classification.value
            segment_series[segment_name].append(uid)
            segment_metrics[segment_name].append(series_metrics[uid].metrics)
    else:
        for uid, series_metric in series_metrics.items():
            segment_series["unknown"].append(uid)
            segment_metrics["unknown"].append(series_metric.metrics)

    result: dict[str, SegmentMetrics] = {}
    for segment_name, series_ids in segment_series.items():
        metrics_list = segment_metrics[segment_name]
        if not metrics_list:
            continue

        aggregated: dict[str, float] = {}
        metric_names: set[str] = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if not np.isnan(m.get(metric_name, np.nan))]
            if values:
                aggregated[metric_name] = float(np.mean(values))

        result[segment_name] = SegmentMetrics(
            segment_name=segment_name,
            series_ids=series_ids,
            metrics=aggregated,
            n_series=len(series_ids),
        )

    return result


def compute_temporal_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]],
    df: pd.DataFrame,
) -> dict[str, TemporalMetrics]:
    """Compute temporal metrics grouped by time dimensions."""
    result: dict[str, TemporalMetrics] = {}

    metric_name = _select_primary_metric(series_metrics_agg)
    if metric_name is None:
        return result

    working = df.copy()
    working["ds"] = pd.to_datetime(working["ds"])

    working["hour"] = working["ds"].dt.hour
    hour_metrics: dict[str, dict[str, float]] = {}
    for hour in sorted(working["hour"].unique()):
        hour_str = str(hour)
        hour_series = working[working["hour"] == hour]["unique_id"].unique()
        if len(hour_series) > 0:
            values = []
            for uid in hour_series:
                if uid in series_metrics_agg and series_metrics_agg[uid]:
                    avg_metric = np.mean([m.get(metric_name, np.nan) for m in series_metrics_agg[uid]])
                    if not np.isnan(avg_metric):
                        values.append(avg_metric)
            if values:
                hour_metrics[hour_str] = {metric_name: float(np.mean(values))}

    if hour_metrics:
        result["hour"] = TemporalMetrics(dimension="hour", metrics_by_value=hour_metrics)

    working["dayofweek"] = working["ds"].dt.dayofweek
    dow_metrics: dict[str, dict[str, float]] = {}
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for dow in sorted(working["dayofweek"].unique()):
        dow_str = dow_names[dow]
        dow_series = working[working["dayofweek"] == dow]["unique_id"].unique()
        if len(dow_series) > 0:
            values = []
            for uid in dow_series:
                if uid in series_metrics_agg and series_metrics_agg[uid]:
                    avg_metric = np.mean([m.get(metric_name, np.nan) for m in series_metrics_agg[uid]])
                    if not np.isnan(avg_metric):
                        values.append(avg_metric)
            if values:
                dow_metrics[dow_str] = {metric_name: float(np.mean(values))}

    if dow_metrics:
        result["dayofweek"] = TemporalMetrics(dimension="dayofweek", metrics_by_value=dow_metrics)

    return result


def _select_primary_metric(
    series_metrics_agg: dict[str, list[dict[str, float]]],
    preferred: tuple[str, ...] = ("wape", "mae", "rmse", "smape", "mase"),
) -> str | None:
    metric_names: set[str] = set()
    for metrics_list in series_metrics_agg.values():
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
    for name in preferred:
        if name in metric_names:
            return name
    return next(iter(metric_names), None)


__all__ = [
    "aggregate_metrics",
    "build_series_metrics",
    "compute_segment_metrics",
    "compute_temporal_metrics",
    "series_metrics_from_frame",
    "summary_to_metrics",
]

