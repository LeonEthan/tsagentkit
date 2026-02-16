"""Moirai 2.0 TSFM adapter for tsagentkit.

Wraps Salesforce's Moirai 2.0 model for zero-shot time series forecasting.
Uses tsagentkit-uni2ts library (pip install tsagentkit-uni2ts) with Salesforce/moirai-2.0-R-small model.
Moirai 2.0 uses a decoder-only architecture (different from Moirai 1.x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


# Module-level cache for the loaded model
_loaded_model: Any | None = None
_default_model_name: str = "Salesforce/moirai-2.0-R-small"


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
    global _loaded_model, _default_model_name

    if _loaded_model is None or _default_model_name != model_name:
        # Import from tsagentkit-uni2ts package (Moirai 2.0 uses moirai2 module)
        from tsagentkit_uni2ts.model.moirai2 import Moirai2Module

        # Load the pretrained module
        module = Moirai2Module.from_pretrained(model_name)

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
    """Generate forecasts using Moirai 2.0.

    Args:
        model: Loaded Moirai model (dict with model_name and module)
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np
    from tsagentkit_uni2ts.model.moirai2 import Moirai2Forecast

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
        ctx_len = len(context)

        # Create forecast model with Moirai 2.0
        # Moirai 2.0 uses target_dim=1 for univariate forecasting
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

        # Prepare data for GluonTS format
        ts_df = pd.DataFrame({
            "target": context,
        }, index=pd.date_range(start=series_df[time_col].iloc[0], periods=ctx_len, freq=dataset.config.freq))

        gts_dataset = PandasDataset([{"target": ts_df["target"].values, "start": ts_df.index[0]}])

        # Generate predictions
        forecast_it = predictor.predict(gts_dataset)
        forecast = next(forecast_it)

        # Extract median forecast (quantile 0.5)
        forecast_values = forecast.quantile(0.5)

        # Generate future timestamps
        last_date = series_df[time_col].iloc[-1]
        freq = dataset.config.freq
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
class MoiraiAdapter:
    """Backward-compatible wrapper for Moirai adapter.

    DEPRECATED: Use module-level functions directly:
        from tsagentkit.models.adapters.tsfm.moirai import load, fit, predict
        model = load(model_name="Salesforce/moirai-2.0-R-small")
        artifact = fit(dataset)
        forecast = predict(artifact, dataset, h=7)
    """

    def __init__(self, model_name: str = "Salesforce/moirai-2.0-R-small"):
        self.model_name = model_name

    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load model and return artifact."""
        model = load(self.model_name)
        return {"model": model["module"], "model_name": self.model_name, "adapter": self}

    def predict(self, dataset: TSDataset, artifact: dict[str, Any], h: int) -> pd.DataFrame:
        """Generate forecasts."""
        model_module = artifact.get("model", _loaded_model["module"] if _loaded_model else None)
        if model_module is None:
            model_module = load(self.model_name)["module"]
        # Reconstruct model dict for predict function
        model = {"model_name": self.model_name, "module": model_module}
        return predict(model, dataset, h)
