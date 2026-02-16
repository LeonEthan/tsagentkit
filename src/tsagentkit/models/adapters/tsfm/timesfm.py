"""TimesFM 2.5 TSFM adapter for tsagentkit.

Wraps Google's TimesFM 2.5 model for zero-shot time series forecasting.
Uses tsagentkit-timesfm library (pip install tsagentkit-timesfm) with google/timesfm-2.5-200m-pytorch model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


# Module-level cache for the loaded model
_loaded_model: Any | None = None


def load() -> Any:
    """Load pretrained TimesFM 2.5 200M model.

    Uses tsagentkit-timesfm library with google/timesfm-2.5-200m-pytorch model.
    Install: pip install tsagentkit-timesfm

    Returns:
        Loaded TimesFM model
    """
    global _loaded_model

    if _loaded_model is None:
        import tsagentkit_timesfm as timesfm

        # Load TimesFM 2.5 200M model using from_pretrained
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )

        # Configure the model for inference
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
            )
        )

        _loaded_model = model

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

    # Map pandas freq to TimesFM frequency token (0=high/daily, 1=medium, 2=low)
    freq_map = {
        "D": 0,
        "H": 0,
        "T": 0,
        "min": 0,
        "B": 0,
        "W": 1,
        "M": 1,
        "MS": 1,
        "Q": 2,
        "QS": 2,
        "Y": 2,
    }
    tfm_freq = freq_map.get(freq, 0)

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].values.astype(np.float32)

        # Generate forecast
        point_forecast, _ = model.forecast(
            horizon=h,
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


# Backward compatibility wrapper
class TimesFMAdapter:
    """Backward-compatible wrapper for TimesFM adapter.

    DEPRECATED: Use module-level functions directly:
        from tsagentkit.models.adapters.tsfm.timesfm import load, fit, predict
        model = load()
        artifact = fit(dataset)
        forecast = predict(artifact, dataset, h=7)
    """

    def __init__(self, context_len: int = 512, horizon_len: int = 128):
        self.context_len = context_len
        self.horizon_len = horizon_len

    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load model and return artifact."""
        model = load()
        return {"model": model, "adapter": self}

    def predict(self, dataset: TSDataset, artifact: dict[str, Any], h: int) -> pd.DataFrame:
        """Generate forecasts."""
        model = artifact.get("model", _loaded_model)
        if model is None:
            model = load()
        return predict(model, dataset, h)
