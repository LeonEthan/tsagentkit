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
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Element-wise ensemble with quantile aggregation.

    Args:
        predictions: List of DataFrames with columns [unique_id, ds, yhat, q0.1, ...]
        method: Aggregation method ("median" or "mean")
        quantiles: List of quantile levels to ensemble

    Returns:
        DataFrame with ensemble predictions and quantiles
    """
    result = ensemble(predictions, method)

    # Ensemble quantile columns if present
    if quantiles:
        import numpy as np

        for q in quantiles:
            q_col = f"q{q}"
            # Check if quantile column exists in at least one prediction
            if any(q_col in p.columns for p in predictions):
                # Stack available quantile values, using yhat as fallback
                q_stack = np.stack([
                    p[q_col].values if q_col in p.columns else p["yhat"].values
                    for p in predictions
                ])
                if method == "median":
                    result[q_col] = np.median(q_stack, axis=0)
                else:
                    result[q_col] = np.mean(q_stack, axis=0)

    return result


__all__ = ["ensemble", "ensemble_with_quantiles"]
