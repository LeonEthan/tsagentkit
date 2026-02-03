"""Evaluation utilities for forecast metrics and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

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
        from utilsforecast import evaluation as ufeval
        from utilsforecast import losses as uflosses
    except Exception:
        return None, None
    return ufeval, uflosses


def _wide_predictions(
    df: pd.DataFrame,
    id_col: str,
    ds_col: str,
    target_col: str,
    model_col: str,
    pred_col: str,
    cutoff_col: str | None,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[float, str]]]:
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
    quantile_cols = [c for c in df.columns if parse_quantile_column(c) is not None]
    quantile_map: dict[str, dict[float, str]] = {}

    if quantile_cols:
        for q_col in quantile_cols:
            q = parse_quantile_column(q_col)
            if q is None:
                continue
            q_pivot = df.pivot_table(
                index=index_cols,
                columns=model_col,
                values=q_col,
                aggfunc="mean",
            ).reset_index()
            rename_map: dict[str, str] = {}
            for col in q_pivot.columns:
                if col in index_cols:
                    continue
                new_col = f"{col}__{q_col}"
                rename_map[col] = new_col
                quantile_map.setdefault(col, {})[q] = new_col
            if rename_map:
                q_pivot = q_pivot.rename(columns=rename_map)
                wide = wide.merge(q_pivot, on=index_cols, how="left")

    return wide, model_cols, quantile_map


def _wrap_metric_name(func: Any, name: str) -> Any:
    func.__name__ = name
    return func


def _make_wape_metric(uflosses: Any, cutoff_col: str) -> Any:
    def _metric(
        df: pd.DataFrame,
        models: list[str],
        id_col: str = "unique_id",
        target_col: str = "y",
        **_: Any,
    ) -> pd.DataFrame:
        return uflosses.nd(
            df=df,
            models=models,
            id_col=id_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
        )

    return _wrap_metric_name(_metric, "wape")


def _make_quantile_loss_metric(
    uflosses: Any,
    q: float,
    quantile_models: dict[str, str],
    cutoff_col: str,
) -> Any:
    def _metric(
        df: pd.DataFrame,
        models: list[str],
        id_col: str = "unique_id",
        target_col: str = "y",
        **_: Any,
    ) -> pd.DataFrame:
        return uflosses.quantile_loss(
            df=df,
            models=quantile_models,
            q=q,
            id_col=id_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
        )

    return _wrap_metric_name(_metric, f"pinball_{q:.3f}")


def _make_wql_metric(
    uflosses: Any,
    quantile_models: dict[str, list[str]],
    quantiles: np.ndarray,
    cutoff_col: str,
) -> Any:
    def _metric(
        df: pd.DataFrame,
        models: list[str],
        id_col: str = "unique_id",
        target_col: str = "y",
        **_: Any,
    ) -> pd.DataFrame:
        return uflosses.mqloss(
            df=df,
            models=quantile_models,
            quantiles=quantiles,
            id_col=id_col,
            target_col=target_col,
            cutoff_col=cutoff_col,
        )

    return _wrap_metric_name(_metric, "wql")


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

    wide, model_cols, quantile_map = _wide_predictions(
        df,
        id_col=id_col,
        ds_col=ds_col,
        target_col=target_col,
        model_col=model_col,
        pred_col=pred_col,
        cutoff_col=cutoff_col,
    )

    ufeval, uflosses = _maybe_import_utilsforecast()
    if ufeval is None or uflosses is None or not model_cols:
        return MetricFrame(pd.DataFrame()), ScoreSummary(pd.DataFrame())

    cutoff_present = cutoff_col is not None and cutoff_col in wide.columns
    cutoff_name = cutoff_col if cutoff_col is not None else "cutoff"
    wide_eval = wide

    metrics: list[Any] = [uflosses.mae, uflosses.rmse, uflosses.smape]
    if hasattr(uflosses, "nd"):
        metrics.append(_make_wape_metric(uflosses, cutoff_name))

    if train_df is not None and season_length and hasattr(uflosses, "mase"):
        mase_metric = partial(uflosses.mase, seasonality=season_length)
        metrics.append(_wrap_metric_name(mase_metric, "mase"))

    if quantile_map and hasattr(uflosses, "quantile_loss"):
        available_models = [model for model in model_cols if model in quantile_map]
        if available_models:
            common_quantiles = set.intersection(
                *[
                    set(quantile_map[model].keys())
                    for model in available_models
                ]
            )
        else:
            common_quantiles = set()

        if common_quantiles:
            quantiles_sorted = sorted(common_quantiles)
            for q in quantiles_sorted:
                per_q_models = {
                    model: quantile_map[model][q]
                    for model in available_models
                    if q in quantile_map[model]
                }
                if per_q_models:
                    metrics.append(
                        _make_quantile_loss_metric(
                            uflosses=uflosses,
                            q=q,
                            quantile_models=per_q_models,
                            cutoff_col=cutoff_name,
                        )
                    )

            per_model_quantiles = {
                model: [quantile_map[model][q] for q in quantiles_sorted]
                for model in available_models
                if all(q in quantile_map[model] for q in quantiles_sorted)
            }
            if per_model_quantiles and hasattr(uflosses, "mqloss"):
                metrics.append(
                    _make_wql_metric(
                        uflosses=uflosses,
                        quantile_models=per_model_quantiles,
                        quantiles=np.asarray(quantiles_sorted, dtype=float),
                        cutoff_col=cutoff_name,
                    )
                )

    metric_df = ufeval.evaluate(
        wide_eval,
        metrics=metrics,
        models=model_cols,
        train_df=train_df,
        id_col=id_col,
        time_col=ds_col,
        target_col=target_col,
        cutoff_col=cutoff_name,
    )

    metrics_long = metric_df.copy()
    index_cols = [id_col]
    if cutoff_present and cutoff_col and cutoff_col in metrics_long.columns:
        index_cols.append(cutoff_col)

    metric_cols = [c for c in metrics_long.columns if c not in index_cols + ["metric"]]
    metrics_long = metrics_long.melt(
        id_vars=index_cols + ["metric"],
        value_vars=metric_cols,
        var_name="model",
        value_name="value",
    )

    summary = (
        metrics_long.groupby(["model", "metric"])["value"]
        .mean()
        .reset_index()
    )

    return MetricFrame(metrics_long), ScoreSummary(summary)


__all__ = ["MetricFrame", "ScoreSummary", "evaluate_forecasts"]
