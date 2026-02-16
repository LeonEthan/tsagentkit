"""TimesFM 2.5 TSFM adapter for tsagentkit.

Wraps Google's TimesFM 2.5 model for zero-shot time series forecasting.
Uses tsagentkit-timesfm library and google/timesfm-2.5-200m-pytorch model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


# Module-level cache for the loaded model
_loaded_model: Any | None = None


def load() -> Any:
    """Load pretrained TimesFM 2.5 200M model.

    Returns:
        Loaded TimesFM model
    """
    global _loaded_model

    if _loaded_model is None:
        import tsagentkit_timesfm as timesfm

        # Load TimesFM 2.5 200M model using from_pretrained
        _loaded_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )

    return _loaded_model


def unload() -> None:
    """Unload model to free memory."""
    global _loaded_model
    _loaded_model = None


def fit(dataset: TSDataset) -> Any:
    """Fit TimesFM model (loads pretrained).

    TimesFM is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate forecasts using TimesFM.

    Args:
        model: Loaded TimesFM model
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np

    forecasts = []
    id_col = dataset.config.id_col
    time_col = dataset.config.time_col
    target_col = dataset.config.target_col
    freq = dataset.config.freq

    # Map pandas freq to TimesFM frequency token
    freq_map = {
        "D": "D",
        "H": "H",
        "M": "M",
        "MS": "M",
        "Q": "Q",
        "QS": "Q",
        "W": "W",
        "B": "B",
    }
    tfm_freq = freq_map.get(freq, "D")

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].values

        # Generate forecast
        point_forecast, _ = model.forecast(
            inputs=[context],
            freq=[tfm_freq],
        )

        # Extract forecast for requested horizon
        forecast_values = point_forecast[0][:h]

        # Generate future timestamps
        last_date = series_df[time_col].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            id_col: unique_id,
            time_col: future_dates[:len(forecast_values)],
            "yhat": forecast_values,
        })
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
