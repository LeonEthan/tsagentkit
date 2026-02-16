"""Moirai 2.0 TSFM adapter for tsagentkit.

Wraps Salesforce's Moirai 2.0 model for zero-shot time series forecasting.
Uses uni2ts library (via tsagentkit-uni2ts) and Salesforce/moirai-2.0-R-small model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


# Module-level cache for the loaded model
_loaded_model: Any | None = None
_default_model_name: str = "Salesforce/moirai-2.0-R-small"


def load(model_name: str = "Salesforce/moirai-2.0-R-small") -> Any:
    """Load pretrained Moirai 2.0 model.

    Args:
        model_name: Moirai model variant (Salesforce/moirai-2.0-R-small, etc.)

    Returns:
        Loaded Moirai model module
    """
    global _loaded_model, _default_model_name

    if _loaded_model is None or _default_model_name != model_name:
        # Import from tsagentkit_uni2ts package
        from tsagentkit_uni2ts.model.moirai import MoiraiModule

        # Load the pretrained module
        module = MoiraiModule.from_pretrained(model_name)

        _loaded_model = {"model_name": model_name, "module": module}
        _default_model_name = model_name

    return _loaded_model


def unload() -> None:
    """Unload model to free memory."""
    global _loaded_model
    _loaded_model = None


def fit(dataset: TSDataset) -> Any:
    """Fit Moirai model (loads pretrained).

    Moirai is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate forecasts using Moirai.

    Args:
        model: Loaded Moirai model (dict with model_name and module)
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np
    from tsagentkit_uni2ts.model.moirai import MoiraiForecast

    model_name = model["model_name"]
    module = model["module"]
    forecasts = []
    id_col = dataset.config.id_col
    time_col = dataset.config.time_col
    target_col = dataset.config.target_col

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].values.astype(np.float32)

        # Create forecast model with Moirai 2.0
        # Moirai 2.0 uses specific patch sizes - auto-select based on context length
        ctx_len = len(context)
        if ctx_len >= 512:
            patch_size = 64
        elif ctx_len >= 128:
            patch_size = 32
        else:
            patch_size = 16

        forecast_model = MoiraiForecast(
            module=module,
            prediction_length=h,
            context_length=ctx_len,
            patch_size=patch_size,
            num_samples=100,
        )

        # Generate forecast - Moirai returns distribution samples
        samples = forecast_model.predict(context, h)

        # Extract median forecast
        forecast_values = np.median(samples, axis=0)

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
