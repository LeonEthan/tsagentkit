"""Evaluation utilities for forecast metrics and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tsagentkit.backtest.metrics import pinball_loss, wql
from tsagentkit.utils import parse_quantile_column


@dataclass(frozen=True)
class MetricFrame:
    """Container for metric results."""

    df: pd.DataFrame


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregate metric summary."""

    df: pd.DataFrame


def _maybe_import_utilsforecast():
    try:
        from utilsforecast import losses as uflosses
    except Exception:
        return None
    return uflosses


def _loss_kwargs(fn, id_col: str, target_col: str, cutoff_col: str | None) -> dict[str, Any]:
    import inspect

    params = inspect.signature(fn).parameters
    kwargs = {"id_col": id_col, "target_col": target_col}
    if cutoff_col and "cutoff_col" in params:
        kwargs["cutoff_col"] = cutoff_col
    return kwargs


def _wide_predictions(
    df: pd.DataFrame,
    id_col: str,
    ds_col: str,
    target_col: str,
    model_col: str,
    pred_col: str,
    cutoff_col: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    index_cols = [id_col, ds_col]
    if cutoff_col and cutoff_col in df.columns:
        index_cols.append(cutoff_col)
    pivot = df.pivot_table(
        index=index_cols,
        columns=model_col,
        values=pred_col,
        aggfunc="mean",
    )
    wide = pivot.reset_index()
    actuals = df[index_cols + [target_col]].drop_duplicates(subset=index_cols)
    wide = wide.merge(actuals, on=index_cols, how="left")
    model_cols = [c for c in wide.columns if c not in index_cols + [target_col]]
    return wide, model_cols


def evaluate_forecasts(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    season_length: int | None = None,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    target_col: str = "y",
    model_col: str = "model",
    pred_col: str = "yhat",
    cutoff_col: str | None = "cutoff",
) -> tuple[MetricFrame, ScoreSummary]:
    """Compute point + quantile metrics in a stable long schema."""
    if df.empty:
        return MetricFrame(pd.DataFrame()), ScoreSummary(pd.DataFrame())

    if model_col not in df.columns:
        df = df.copy()
        df[model_col] = "model"

    wide, model_cols = _wide_predictions(
        df,
        id_col=id_col,
        ds_col=ds_col,
        target_col=target_col,
        model_col=model_col,
        pred_col=pred_col,
        cutoff_col=cutoff_col,
    )

    uflosses = _maybe_import_utilsforecast()
    metrics_frames: list[pd.DataFrame] = []

    if uflosses is not None:
        for name, fn in [
            ("mae", uflosses.mae),
            ("rmse", uflosses.rmse),
            ("smape", uflosses.smape),
        ]:
            kwargs = _loss_kwargs(fn, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col)
            metric_df = fn(wide, model_cols=model_cols, **kwargs)
            metric_df["metric"] = name
            metrics_frames.append(metric_df)

        if train_df is not None and season_length:
            kwargs = _loss_kwargs(
                uflosses.mase,
                id_col=id_col,
                target_col=target_col,
                cutoff_col=cutoff_col,
            )
            metric_df = uflosses.mase(
                wide,
                train_df=train_df,
                season_length=season_length,
                model_cols=model_cols,
                **kwargs,
            )
            metric_df["metric"] = "mase"
            metrics_frames.append(metric_df)

    if not metrics_frames:
        return MetricFrame(pd.DataFrame()), ScoreSummary(pd.DataFrame())

    metrics_long = pd.concat(metrics_frames, ignore_index=True)
    index_cols = [id_col]
    if cutoff_col and cutoff_col in metrics_long.columns:
        index_cols.append(cutoff_col)

    metric_cols = [c for c in metrics_long.columns if c not in index_cols + ["metric"]]
    metrics_long = metrics_long.melt(
        id_vars=index_cols + ["metric"],
        value_vars=metric_cols,
        var_name="model",
        value_name="value",
    )

    quantile_cols = [c for c in df.columns if parse_quantile_column(c) is not None]
    quantile_records: list[dict[str, Any]] = []

    if quantile_cols:
        group_cols = [id_col, model_col]
        if cutoff_col and cutoff_col in df.columns:
            group_cols.append(cutoff_col)
        for keys, group in df.groupby(group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            record_base = dict(zip(group_cols, keys))
            y_true = group[target_col].to_numpy(dtype=float)
            q_map: dict[float, np.ndarray] = {}
            for col in quantile_cols:
                q = parse_quantile_column(col)
                if q is None:
                    continue
                q_map[q] = group[col].to_numpy(dtype=float)
                quantile_records.append({
                    **record_base,
                    "metric": f"pinball_{q:.3f}",
                    "value": float(pinball_loss(y_true, q_map[q], q)),
                })
            if q_map:
                quantile_records.append({
                    **record_base,
                    "metric": "wql",
                    "value": float(wql(y_true, q_map)),
                })

    if quantile_records:
        quantile_df = pd.DataFrame(quantile_records)
        quantile_df = quantile_df.rename(columns={model_col: "model"})
        metrics_long = pd.concat([metrics_long, quantile_df], ignore_index=True)

    summary = (
        metrics_long.groupby(["model", "metric"])["value"]
        .mean()
        .reset_index()
    )

    return MetricFrame(metrics_long), ScoreSummary(summary)


__all__ = ["MetricFrame", "ScoreSummary", "evaluate_forecasts"]
