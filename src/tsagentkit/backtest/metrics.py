"""Metrics calculation for time series forecasting.

Deprecated: use ``tsagentkit.eval.evaluate_forecasts`` (utilsforecast-backed)
for new evaluation paths. This module is retained for backward compatibility
and will be removed in a future phase.
"""

from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd

from tsagentkit.eval import evaluate_forecasts
from tsagentkit.utils import quantile_col_name

_DEPRECATION_MESSAGE = (
    "tsagentkit.backtest.metrics is deprecated; "
    "use tsagentkit.eval.evaluate_forecasts instead."
)


def _warn_deprecated(name: str) -> None:
    warnings.warn(
        f"{name} is deprecated. {_DEPRECATION_MESSAGE}",
        DeprecationWarning,
        stacklevel=2,
    )


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _build_eval_frame(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    y_quantiles: dict[float, Iterable[float]] | None = None,
) -> pd.DataFrame:
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    n = y_true_arr.shape[0]
    df = pd.DataFrame(
        {
            "unique_id": ["series"] * n,
            "ds": pd.date_range("2000-01-01", periods=n, freq="D"),
            "y": y_true_arr,
            "yhat": y_pred_arr,
            "model": "model",
        }
    )

    if y_quantiles:
        for q, values in y_quantiles.items():
            q_values = _as_array(values)
            if q_values.shape != y_true_arr.shape:
                raise ValueError("Quantile predictions must match y_true shape.")
            df[quantile_col_name(float(q))] = q_values

    return df


def _build_train_frame(
    y_train: Iterable[float],
) -> pd.DataFrame:
    y_train_arr = _as_array(y_train)
    n = y_train_arr.shape[0]
    return pd.DataFrame(
        {
            "unique_id": ["series"] * n,
            "ds": pd.date_range("1999-01-01", periods=n, freq="D"),
            "y": y_train_arr,
        }
    )


def _summary_to_metrics(summary_df: pd.DataFrame, model_name: str = "model") -> dict[str, float]:
    if summary_df.empty:
        return {}
    df = summary_df
    if "model" in df.columns:
        df = df[df["model"] == model_name]
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return {}
    return {row["metric"]: float(row["value"]) for _, row in df.iterrows()}


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Weighted Absolute Percentage Error (utilsforecast ND)."""
    _warn_deprecated("wape")
    df = _build_eval_frame(y_true, y_pred)
    _, summary = evaluate_forecasts(df)
    metrics = _summary_to_metrics(summary.df)
    return float(metrics.get("wape", np.nan))


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    _warn_deprecated("smape")
    df = _build_eval_frame(y_true, y_pred)
    _, summary = evaluate_forecasts(df)
    metrics = _summary_to_metrics(summary.df)
    return float(metrics.get("smape", np.nan))


def mase(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    y_train: Iterable[float],
    season_length: int = 1,
) -> float:
    """Mean Absolute Scaled Error."""
    _warn_deprecated("mase")
    df = _build_eval_frame(y_true, y_pred)
    train_df = _build_train_frame(y_train)
    _, summary = evaluate_forecasts(df, train_df=train_df, season_length=season_length)
    metrics = _summary_to_metrics(summary.df)
    value = float(metrics.get("mase", np.nan))
    if np.isinf(value):
        return float("nan")
    return value


def pinball_loss(
    y_true: Iterable[float],
    y_quantile: Iterable[float],
    tau: float,
) -> float:
    """Pinball Loss for quantile forecasts."""
    _warn_deprecated("pinball_loss")
    df = _build_eval_frame(y_true, y_quantile, y_quantiles={tau: y_quantile})
    _, summary = evaluate_forecasts(df)
    metrics = _summary_to_metrics(summary.df)
    return float(metrics.get(f"pinball_{tau:.3f}", np.nan))


def wql(y_true: Iterable[float], y_quantiles: dict[float, Iterable[float]]) -> float:
    """Weighted Quantile Loss (average pinball loss across quantiles)."""
    _warn_deprecated("wql")
    df = _build_eval_frame(y_true, y_true, y_quantiles=y_quantiles)
    _, summary = evaluate_forecasts(df)
    metrics = _summary_to_metrics(summary.df)
    return float(metrics.get("wql", np.nan))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean Absolute Error."""
    _warn_deprecated("mae")
    df = _build_eval_frame(y_true, y_pred)
    _, summary = evaluate_forecasts(df)
    metrics = _summary_to_metrics(summary.df)
    return float(metrics.get("mae", np.nan))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    _warn_deprecated("rmse")
    df = _build_eval_frame(y_true, y_pred)
    _, summary = evaluate_forecasts(df)
    metrics = _summary_to_metrics(summary.df)
    return float(metrics.get("rmse", np.nan))


def compute_all_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    y_train: Iterable[float] | None = None,
    season_length: int = 1,
    y_quantiles: dict[float, Iterable[float]] | None = None,
) -> dict[str, float]:
    """Compute all standard metrics via utilsforecast.evaluate."""
    _warn_deprecated("compute_all_metrics")
    df = _build_eval_frame(y_true, y_pred, y_quantiles=y_quantiles)
    train_df = _build_train_frame(y_train) if y_train is not None else None
    _, summary = evaluate_forecasts(
        df,
        train_df=train_df,
        season_length=season_length if train_df is not None else None,
    )
    metrics = _summary_to_metrics(summary.df)

    if "mase" not in metrics:
        metrics["mase"] = float("nan")

    if y_quantiles:
        for q in y_quantiles:
            source_key = f"pinball_{q:.3f}"
            target_key = f"pinball_{q:.2f}"
            if source_key in metrics and target_key not in metrics:
                metrics[target_key] = metrics[source_key]

    return metrics


def compute_metrics_by_series(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "yhat",
) -> dict[str, dict[str, float]]:
    """Compute metrics for each series separately via utilsforecast.evaluate."""
    _warn_deprecated("compute_metrics_by_series")
    if df.empty:
        return {}

    working = df.copy()
    if "model" not in working.columns:
        working["model"] = "model"
    if "ds" not in working.columns:
        working["ds"] = working.groupby(id_col).cumcount()

    metric_frame, _ = evaluate_forecasts(
        working,
        id_col=id_col,
        ds_col="ds",
        target_col=actual_col,
        model_col="model",
        pred_col=pred_col,
        cutoff_col=None,
    )

    metrics_df = metric_frame.df
    if metrics_df is None or metrics_df.empty:
        return {}

    if "model" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["model"] == "model"]

    if id_col not in metrics_df.columns or "metric" not in metrics_df.columns:
        return {}

    grouped = metrics_df.groupby([id_col, "metric"])["value"].mean().reset_index()
    result: dict[str, dict[str, float]] = {}
    for uid, group in grouped.groupby(id_col):
        result[str(uid)] = {
            row["metric"]: float(row["value"]) for _, row in group.iterrows()
        }

    return result
