"""Moirai 2.0 TSFM adapter for tsagentkit.

Wraps Salesforce's Moirai 2.0 model for zero-shot time series forecasting.
Uses tsagentkit-uni2ts library (pip install tsagentkit-uni2ts) with Salesforce/moirai-2.0-R-small model.
Moirai 2.0 uses a decoder-only architecture (different from Moirai 1.x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.core.dataset import _normalize_freq_alias

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def load(model_name: str = "Salesforce/moirai-2.0-R-small", device: str | None = None) -> Any:
    """Load pretrained Moirai 2.0 model.

    Uses tsagentkit-uni2ts library with Salesforce/moirai-2.0-R-small model.
    Install: pip install tsagentkit-uni2ts

    Moirai 2.0 uses Moirai2Module and Moirai2Forecast (decoder-only architecture).

    Args:
        model_name: Moirai model variant (Salesforce/moirai-2.0-R-small, etc.)
        device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        Loaded Moirai model module
    """
    from uni2ts.model.moirai2 import Moirai2Module

    from tsagentkit.core.device import resolve_device

    resolved = resolve_device(device or "auto", allow_mps=True)

    # Load the pretrained module
    module = Moirai2Module.from_pretrained(model_name)

    # Move module to device if supported
    if hasattr(module, "to") and resolved in ("cuda", "mps", "cpu"):
        import torch

        module = module.to(torch.device(resolved))

    return {"model_name": model_name, "module": module}


def unload(model: Any | None = None) -> None:
    """Unload model resources (best-effort)."""
    del model


def fit(dataset: TSDataset) -> Any:
    """Fit Moirai model (loads pretrained).

    Moirai is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(
    model: Any, dataset: TSDataset, h: int, batch_size: int = 32
) -> pd.DataFrame:
    """Generate forecasts using Moirai 2.0.

    Args:
        model: Loaded Moirai model (dict with model_name and module)
        dataset: Time-series dataset
        h: Forecast horizon
        batch_size: Number of series to process in parallel

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai2 import Moirai2Forecast

    module = model["module"]
    freq = _normalize_freq_alias(dataset.config.freq)

    # Group data by unique_id (dataset is pre-sorted by TSDataset)
    grouped = dataset.df.groupby("unique_id", sort=False)

    # Pre-extract all series data
    series_data = []
    for unique_id, group in grouped:
        context = group["y"].values.astype(np.float32)
        last_date = group["ds"].iloc[-1]
        start_date = group["ds"].iloc[0]
        series_data.append((unique_id, context, last_date, start_date))

    forecasts = []
    # Process in batches
    for i in range(0, len(series_data), batch_size):
        batch = series_data[i : i + batch_size]

        # Determine max context length for this batch
        max_ctx_len = max(len(ctx) for _, ctx, _, _ in batch)

        # Create forecast model with batch-aware context length
        forecast_model = Moirai2Forecast(
            module=module,
            prediction_length=h,
            context_length=max_ctx_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # Use batch_size > 1 for efficient inference
        predictor = forecast_model.create_predictor(batch_size=len(batch))

        # Build multi-series PandasDataset
        ts_dict = {}
        for unique_id, context, _, start_date in batch:
            ts_index = pd.date_range(start=start_date, periods=len(context), freq=freq)
            ts_dict[unique_id] = pd.Series(context, index=ts_index)

        # PandasDataset with multiple series
        gts_dataset = PandasDataset(ts_dict)

        # Batch inference
        forecast_it = predictor.predict(gts_dataset)

        # Extract forecasts for each series
        for (unique_id, _, last_date, _), forecast in zip(batch, forecast_it):
            forecast_values = forecast.quantile(0.5)
            future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame({
                "unique_id": unique_id,
                "ds": future_dates[: len(forecast_values)],
                "yhat": forecast_values,
            })
            forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
