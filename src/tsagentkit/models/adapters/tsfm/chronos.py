"""Chronos2 TSFM adapter for tsagentkit.

Wraps Amazon's Chronos2 model for zero-shot time series forecasting.
Uses chronos library (pip install chronos-forecasting) with amazon/chronos-2 model.
Uses module-level caching for efficient model reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


# Module-level cache for the loaded model
# ModelCache manages the lifecycle, this is just the storage
_loaded_model: Any | None = None
_default_model_name: str = "amazon/chronos-2"


def load(model_name: str = "amazon/chronos-2") -> Any:
    """Load pretrained Chronos2 model.

    Uses chronos library (https://github.com/amazon-science/chronos-forecasting)
    Install: pip install chronos-forecasting pandas pyarrow

    Args:
        model_name: Chronos model version (amazon/chronos-2, amazon/chronos-2-small, etc.)

    Returns:
        Loaded Chronos2 pipeline
    """
    global _loaded_model, _default_model_name

    if _loaded_model is None or _default_model_name != model_name:
        # chronos package import (from chronos-forecasting)
        from chronos import Chronos2Pipeline

        _loaded_model = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map="auto",
        )
        _default_model_name = model_name

    return _loaded_model


def unload() -> None:
    """Unload model to free memory."""
    global _loaded_model
    _loaded_model = None


def fit(dataset: TSDataset) -> Any:
    """Fit Chronos model (loads pretrained).

    Chronos is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate forecasts using Chronos.

    Args:
        model: Loaded Chronos2 pipeline
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import torch

    forecasts = []
    id_col = dataset.config.id_col
    time_col = dataset.config.time_col
    target_col = dataset.config.target_col

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].values

        # Convert to tensor - Chronos 2 expects 3-d shape (n_series, n_variates, history_length)
        context_tensor = torch.tensor(context, dtype=torch.float32)
        # Reshape from (history_length,) to (1, 1, history_length) for univariate single series
        context_tensor = context_tensor.unsqueeze(0).unsqueeze(0)

        # Generate forecast
        with torch.no_grad():
            prediction = model.predict(context_tensor, h)

        # prediction is a list of tensors, each with shape (n_samples, n_variates, prediction_length)
        # For univariate: extract first element, take median across samples (dim=0), get first variate
        pred_tensor = prediction[0]  # Shape: (n_samples, n_variates, prediction_length)
        forecast_values = pred_tensor.median(dim=0).values[0].numpy()  # Take median across samples, first variate

        # Generate future timestamps
        last_date = series_df[time_col].iloc[-1]
        freq = dataset.config.freq
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            id_col: unique_id,
            time_col: future_dates,
            "yhat": forecast_values,
        })
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)


