"""Metrics calculation for time series forecasting.

Implements common forecasting metrics: WAPE, SMAPE, MASE, Pinball Loss, WQL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    Also known as MAPE (Mean Absolute Percentage Error) but weighted
    by the sum of actuals rather than per-observation.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        WAPE as a float (0-1 scale, multiply by 100 for percentage)

    Raises:
        ValueError: If sum of y_true is zero
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.sum(np.abs(y_true)) == 0:
        raise ValueError("Cannot compute WAPE: sum of y_true is zero")

    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    A symmetric version of MAPE that handles zero values better.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        SMAPE as a float (0-1 scale, multiply by 100 for percentage)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales the forecast error by the in-sample MAE of the naive
    seasonal forecast.

    Args:
        y_true: Actual values (test set)
        y_pred: Predicted values
        y_train: Training values for scaling
        season_length: Seasonal period for naive forecast

    Returns:
        MASE value (1.0 means same error as naive seasonal)

    Raises:
        ValueError: If naive forecast MAE is zero
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Compute forecast MAE
    mae_forecast = np.mean(np.abs(y_true - y_pred))

    # Compute naive seasonal forecast MAE on training data
    if len(y_train) <= season_length:
        # Not enough data for seasonal naive, use regular naive
        naive_errors = np.abs(y_train[1:] - y_train[:-1])
    else:
        naive_errors = np.abs(y_train[season_length:] - y_train[:-season_length])

    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        raise ValueError("Cannot compute MASE: naive forecast MAE is zero")

    return mae_forecast / mae_naive


def pinball_loss(
    y_true: np.ndarray,
    y_quantile: np.ndarray,
    tau: float,
) -> float:
    """Pinball Loss for quantile forecasts.

    Also known as Quantile Loss. Lower is better.

    Args:
        y_true: Actual values
        y_quantile: Predicted quantile values
        tau: Quantile level (0-1)

    Returns:
        Average pinball loss
    """
    y_true = np.asarray(y_true)
    y_quantile = np.asarray(y_quantile)

    errors = y_true - y_quantile
    loss = np.where(errors >= 0, tau * errors, (tau - 1) * errors)

    return np.mean(loss)


def wql(y_true: np.ndarray, y_quantiles: dict[float, np.ndarray]) -> float:
    """Weighted Quantile Loss (average pinball loss across quantiles)."""
    if not y_quantiles:
        return float("nan")
    losses = []
    for q, preds in y_quantiles.items():
        losses.append(pinball_loss(y_true, preds, q))
    return float(np.mean(losses)) if losses else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    season_length: int = 1,
    y_quantiles: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute all standard metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values (point forecast)
        y_train: Training values for MASE
        season_length: Seasonal period for MASE
        y_quantiles: Optional dict of {quantile: predictions}

    Returns:
        Dictionary of metric name to value
    """
    metrics: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

    # WAPE
    try:
        metrics["wape"] = wape(y_true, y_pred)
    except ValueError:
        metrics["wape"] = np.nan

    # SMAPE
    metrics["smape"] = smape(y_true, y_pred)

    # MASE
    if y_train is not None and len(y_train) > 0:
        try:
            metrics["mase"] = mase(y_true, y_pred, y_train, season_length)
        except ValueError:
            metrics["mase"] = np.nan
    else:
        metrics["mase"] = np.nan

    # Pinball loss for each quantile
    if y_quantiles:
        for q, y_q in y_quantiles.items():
            key = f"pinball_{q:.2f}"
            metrics[key] = pinball_loss(y_true, y_q, q)
        metrics["wql"] = wql(y_true, y_quantiles)

    return metrics


def compute_metrics_by_series(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "yhat",
) -> dict[str, dict[str, float]]:
    """Compute metrics for each series separately.

    Args:
        df: DataFrame with actuals and predictions
        id_col: Column name for series identifier
        actual_col: Column name for actual values
        pred_col: Column name for predicted values

    Returns:
        Dict mapping series_id to metrics dict
    """
    results: dict[str, dict[str, float]] = {}

    for uid in df[id_col].unique():
        series_df = df[df[id_col] == uid]
        y_true = series_df[actual_col].values
        y_pred = series_df[pred_col].values

        metrics: dict[str, float] = {
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        }

        try:
            metrics["wape"] = wape(y_true, y_pred)
        except ValueError:
            metrics["wape"] = np.nan

        metrics["smape"] = smape(y_true, y_pred)

        results[uid] = metrics

    return results
