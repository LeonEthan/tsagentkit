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
        import timesfm

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

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].values.astype(np.float32)

        # TimesFM 2.5 requires sequences > 992 tokens to avoid NaN output
        # See: https://github.com/google-research/timesfm/issues/321
        min_context_length = 993
        if len(context) < min_context_length:
            # Pad with zeros at the beginning
            padding_needed = min_context_length - len(context)
            context = np.pad(context, (padding_needed, 0), mode='constant', constant_values=0)

        # Generate forecast (freq parameter not supported in this TimesFM version)
        point_forecast, _ = model.forecast(
            horizon=h,
            inputs=[context],
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


