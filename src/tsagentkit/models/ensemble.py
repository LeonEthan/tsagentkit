"""Ensemble aggregation for multiple model predictions.

Provides median and mean aggregation across multiple forecast DataFrames.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from tsagentkit.core.errors import EInsufficient


def ensemble(
    predictions: list[pd.DataFrame],
    method: Literal["median", "mean"] = "median",
) -> pd.DataFrame:
    """Element-wise ensemble of predictions.

    Args:
        predictions: List of DataFrames with columns [unique_id, ds, yhat]
        method: Aggregation method ("median" or "mean")

    Returns:
        DataFrame with ensemble predictions

    Raises:
        EInsufficient: If no predictions provided
    """
    if not predictions:
        raise EInsufficient("No predictions to ensemble")

    if len(predictions) == 1:
        return predictions[0].copy()

    # Use first prediction as base for structure
    base = predictions[0].copy()

    # Stack yhat values from all predictions
    yhat_stack = np.stack([p["yhat"].values for p in predictions])

    # Compute ensemble
    if method == "median":
        base["yhat"] = np.median(yhat_stack, axis=0)
    elif method == "mean":
        base["yhat"] = np.mean(yhat_stack, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return base


def ensemble_with_quantiles(
    predictions: list[pd.DataFrame],
    method: Literal["median", "mean"] = "median",
    quantiles: tuple[float, ...] | list[float] | None = None,
    quantile_mode: Literal["best_effort", "strict"] = "best_effort",
) -> pd.DataFrame:
    """Element-wise ensemble with quantile aggregation.

    Args:
        predictions: List of DataFrames with columns [unique_id, ds, yhat, q0.1, ...]
        method: Aggregation method ("median" or "mean")
        quantiles: Quantile levels to ensemble (tuple or list)
        quantile_mode: Missing quantile policy. "best_effort" skips missing
            quantiles; "strict" raises if a requested quantile is unavailable.

    Returns:
        DataFrame with ensemble predictions and quantiles

    Raises:
        EInsufficient: If no predictions provided
    """
    if not predictions:
        raise EInsufficient("No predictions to ensemble")

    if len(predictions) == 1:
        return predictions[0].copy()

    result = ensemble(predictions, method)

    # Ensemble quantile columns if present
    if quantiles:
        if quantile_mode not in {"best_effort", "strict"}:
            raise ValueError(
                f"Unknown quantile_mode: {quantile_mode}. Expected 'best_effort' or 'strict'."
            )

        # Convert tuple to list for iteration
        quantiles_list = list(quantiles) if isinstance(quantiles, tuple) else quantiles

        for q in quantiles_list:
            q_col = f"q{q}"
            q_values = [p[q_col].values for p in predictions if q_col in p.columns]
            if not q_values:
                if quantile_mode == "strict":
                    raise EInsufficient(
                        f"Requested quantile '{q_col}' is unavailable from all models",
                    )
                continue

            q_stack = np.stack(q_values)
            if method == "median":
                result[q_col] = np.median(q_stack, axis=0)
            else:
                result[q_col] = np.mean(q_stack, axis=0)

    return result


def ensemble_streaming(
    predictions: list[pd.DataFrame],
    method: Literal["median", "mean"] = "median",
    quantiles: tuple[float, ...] | list[float] | None = None,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Memory-efficient ensemble using chunked processing.

    Processes predictions in chunks to reduce memory usage for large panels.
    Particularly useful when forecasting many series (100k+) where np.stack()
    would create prohibitively large intermediate arrays.

    Args:
        predictions: List of DataFrames with columns [unique_id, ds, yhat, q0.1, ...]
        method: Aggregation method ("median" or "mean")
        quantiles: Quantile levels to ensemble
        chunk_size: Number of rows to process per chunk

    Returns:
        DataFrame with ensemble predictions

    Raises:
        EInsufficient: If no predictions provided

    Examples:
        >>> # For large panels, use streaming to reduce memory
        >>> result = ensemble_streaming(predictions, chunk_size=5000)
    """
    if not predictions:
        raise EInsufficient("No predictions to ensemble")

    if len(predictions) == 1:
        return predictions[0].copy()

    # Use first prediction as base structure
    base = predictions[0].copy()
    n_rows = len(base)

    # Get quantile columns from first prediction that has them
    quantiles_list: list[float] = []
    if quantiles:
        quantiles_list = list(quantiles) if isinstance(quantiles, tuple) else quantiles

    # Get yhat column index for efficient assignment
    yhat_idx = base.columns.get_loc("yhat")

    # Process in chunks
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        # Stack yhat values for this chunk only
        chunk_stack = np.stack([p["yhat"].values[start:end] for p in predictions])

        # Compute ensemble for chunk
        if method == "median":
            base.iloc[start:end, yhat_idx] = np.median(chunk_stack, axis=0)
        else:
            base.iloc[start:end, yhat_idx] = np.mean(chunk_stack, axis=0)

        # Process quantiles for this chunk
        for q in quantiles_list:
            q_col = f"q{q}"
            q_idx = base.columns.get_loc(q_col) if q_col in base.columns else None
            if q_idx is None:
                continue

            q_chunk = [p[q_col].values[start:end] for p in predictions if q_col in p.columns]
            if not q_chunk:
                continue

            q_stack = np.stack(q_chunk)
            if method == "median":
                base.iloc[start:end, q_idx] = np.median(q_stack, axis=0)
            else:
                base.iloc[start:end, q_idx] = np.mean(q_stack, axis=0)

    return base


__all__ = ["ensemble", "ensemble_with_quantiles", "ensemble_streaming"]
