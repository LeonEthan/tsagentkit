"""Chronos2 TSFM adapter for tsagentkit.

Wraps Amazon's Chronos2 model for zero-shot time series forecasting.
Uses chronos library (pip install chronos-forecasting) with amazon/chronos-2 model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tsagentkit.core.dataset import _normalize_freq_alias
from tsagentkit.models.length_utils import adjust_context_length, validate_prediction_length

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset

logger = logging.getLogger(__name__)


def load(model_name: str = "amazon/chronos-2", device: str | None = None) -> Any:
    """Load pretrained Chronos2 model.

    Uses chronos library (https://github.com/amazon-science/chronos-forecasting)
    Install: pip install chronos-forecasting pandas pyarrow

    Args:
        model_name: Chronos model version (amazon/chronos-2, amazon/chronos-2-small, etc.)
        device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        Loaded Chronos2 pipeline
    """
    from chronos import Chronos2Pipeline

    from tsagentkit.core.device import resolve_device

    resolved = resolve_device(device or "auto", allow_mps=True)
    # Chronos uses HF device_map format
    device_map = resolved if resolved in ("cuda", "cpu") else "auto"

    return Chronos2Pipeline.from_pretrained(
        model_name,
        device_map=device_map,
    )


def unload(model: Any | None = None) -> None:
    """Unload model resources (best-effort)."""
    del model


def fit(dataset: TSDataset) -> Any:
    """Fit Chronos model (loads pretrained).

    Chronos is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(
    model: Any,
    dataset: TSDataset,
    h: int,
    batch_size: int = 32,
    quantiles: tuple[float, ...] | list[float] | None = None,
    spec: Any | None = None,
) -> pd.DataFrame:
    """Generate forecasts using Chronos.

    Args:
        model: Loaded Chronos2 pipeline
        dataset: Time-series dataset
        h: Forecast horizon
        batch_size: Number of series to process in parallel
        quantiles: Optional quantile levels to include as q{level} columns
        spec: Optional ModelSpec with length limits (uses registry if not provided)

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import torch

    from tsagentkit.models.registry import get_spec

    if spec is None:
        spec = get_spec("chronos")

    freq = _normalize_freq_alias(dataset.config.freq)
    requested_quantiles = [float(q) for q in quantiles] if quantiles else []

    # Validate horizon against model limits
    h = validate_prediction_length(h, spec, strict=False)

    # Group data by unique_id (dataset is pre-sorted by TSDataset)
    grouped = dataset.df.groupby("unique_id", sort=False)

    # Check if future covariates are available
    has_future_covariates = (
        dataset.covariates is not None
        and not dataset.covariates.is_empty()
        and dataset.covariates.future is not None
    )

    # Get feature columns from future covariates (excluding unique_id and ds)
    feature_cols: list[str] = []
    if (
        has_future_covariates
        and dataset.covariates is not None
        and dataset.covariates.future is not None
    ):
        cov_cols = dataset.covariates.future.columns.tolist()
        feature_cols = [c for c in cov_cols if c not in ("unique_id", "ds")]

    # Pre-extract all series data
    series_data = []
    for unique_id, group in grouped:
        context = group["y"].values.astype(np.float32)
        last_date = group["ds"].iloc[-1]
        series_data.append((unique_id, context, last_date))

    forecasts = []
    # Process in batches
    for i in range(0, len(series_data), batch_size):
        batch = series_data[i : i + batch_size]

        # Pad contexts to same length for batching
        max_len = max(len(ctx) for _, ctx, _ in batch)
        padded_contexts = []
        for _, context, _ in batch:
            if len(context) < max_len:
                # Left-pad with zeros to maintain recency
                padded = np.pad(
                    context, (max_len - len(context), 0), mode="constant", constant_values=0
                )
            else:
                padded = context
            padded_contexts.append(padded)

        # Stack into batch tensor: (batch_size, 1, history_length)
        context_tensor = torch.tensor(np.stack(padded_contexts), dtype=torch.float32).unsqueeze(1)

        # Prepare covariates if available
        feat_dynamic_real = None
        if has_future_covariates and len(feature_cols) > 0:
            batch_covariates = []
            for unique_id, context, _ in batch:
                context_len = len(context)
                # Get future covariates for this series
                future_cov_df = dataset.get_covariates_for_series(unique_id, "future")

                if future_cov_df is not None and len(future_cov_df) >= h:
                    # Extract historical covariates aligned with context
                    # For Chronos, we need covariates for the full context + prediction window
                    # Build covariate tensor: (n_features, context_length + h)
                    cov_values = np.zeros((len(feature_cols), context_len + h), dtype=np.float32)

                    # For historical part, use covariates from history if they exist in main df
                    # For future part, use the future covariates
                    future_cov_values = future_cov_df[feature_cols].values[:h].T  # (n_features, h)
                    cov_values[:, context_len:] = future_cov_values

                    # Pad or trim to match max_len context
                    if cov_values.shape[1] < max_len + h:
                        # Pad on the left
                        pad_width = max_len + h - cov_values.shape[1]
                        cov_values = np.pad(cov_values, ((0, 0), (pad_width, 0)), mode="constant")
                    elif cov_values.shape[1] > max_len + h:
                        # Trim from the left to keep the most recent values
                        cov_values = cov_values[:, -(max_len + h) :]

                    batch_covariates.append(cov_values)
                else:
                    # No covariates for this series, use zeros
                    batch_covariates.append(
                        np.zeros((len(feature_cols), max_len + h), dtype=np.float32)
                    )

            # Stack into batch tensor: (batch_size, n_features, max_len + h)
            feat_dynamic_real = torch.tensor(np.stack(batch_covariates), dtype=torch.float32)

        # Batch inference
        with torch.no_grad():
            if feat_dynamic_real is not None:
                predictions = model.predict(context_tensor, h, feat_dynamic_real=feat_dynamic_real)
            else:
                predictions = model.predict(context_tensor, h)

        # Extract forecasts for each series
        for j, (unique_id, _, last_date) in enumerate(batch):
            # predictions[j] has shape (n_samples, n_variates, prediction_length)
            pred_tensor = predictions[j]
            sample_forecasts = pred_tensor[:, 0, :]
            forecast_values = torch.quantile(sample_forecasts, q=0.5, dim=0).detach().cpu().numpy()

            future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_dates,
                    "yhat": forecast_values,
                }
            )
            for q in requested_quantiles:
                q_values = torch.quantile(sample_forecasts, q=q, dim=0).detach().cpu().numpy()
                forecast_df[f"q{q}"] = q_values
            forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
