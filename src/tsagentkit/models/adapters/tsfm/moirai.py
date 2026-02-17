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


def load(model_name: str = "Salesforce/moirai-2.0-R-small") -> Any:
    """Load pretrained Moirai 2.0 model.

    Uses tsagentkit-uni2ts library with Salesforce/moirai-2.0-R-small model.
    Install: pip install tsagentkit-uni2ts

    Moirai 2.0 uses Moirai2Module and Moirai2Forecast (decoder-only architecture).

    Args:
        model_name: Moirai model variant (Salesforce/moirai-2.0-R-small, etc.)

    Returns:
        Loaded Moirai model module
    """
    # Import from tsagentkit-uni2ts package (Moirai 2.0 uses moirai2 module)
    from uni2ts.model.moirai2 import Moirai2Module

    # Load the pretrained module
    module = Moirai2Module.from_pretrained(model_name)

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


def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate forecasts using Moirai 2.0.

    Args:
        model: Loaded Moirai model (dict with model_name and module)
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np
    from uni2ts.model.moirai2 import Moirai2Forecast

    module = model["module"]
    forecasts = []
    freq = _normalize_freq_alias(dataset.config.freq)

    # Process each series
    for unique_id in dataset.df["unique_id"].unique():
        mask = dataset.df["unique_id"] == unique_id
        series_df = dataset.df[mask].sort_values("ds")
        context = series_df["y"].values.astype(np.float32)
        ctx_len = len(context)

        # Create forecast model with Moirai 2.0
        forecast_model = Moirai2Forecast(
            module=module,
            prediction_length=h,
            context_length=ctx_len,
            target_dim=1,  # Univariate forecasting
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # Generate forecast - Moirai 2.0 returns distribution
        # Use create_predictor for batch inference
        predictor = forecast_model.create_predictor(batch_size=1)

        # Create a simple GluonTS-compatible dataset for prediction
        from gluonts.dataset.pandas import PandasDataset

        # Prepare data for GluonTS format - PandasDataset expects a DataFrame or Series
        ts_index = pd.date_range(start=series_df["ds"].iloc[0], periods=ctx_len, freq=freq)
        ts_series = pd.Series(context, index=ts_index)

        gts_dataset = PandasDataset(ts_series)

        # Generate predictions
        forecast_it = predictor.predict(gts_dataset)
        forecast = next(forecast_it)

        # Extract median forecast (quantile 0.5)
        forecast_values = forecast.quantile(0.5)

        # Generate future timestamps
        last_date = series_df["ds"].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "unique_id": unique_id,
            "ds": future_dates[:len(forecast_values)],
            "yhat": forecast_values,
        })
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
