"""Moirai TSFM adapter for tsagentkit.

Wraps Salesforce's Moirai model for zero-shot time series forecasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


class MoiraiAdapter:
    """Moirai adapter for zero-shot forecasting."""

    def __init__(self, model_name: str = "moirai-1.1-R-small"):
        """Initialize adapter with model specification.

        Args:
            model_name: Moirai model variant (moirai-1.1-R-small/base/large)
        """
        self.model_name = model_name
        self._model = None

    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load pretrained Moirai model.

        Moirai is a zero-shot model, so fit() just loads the model.

        Args:
            dataset: Time-series dataset (used for validation only)

        Returns:
            Model artifact containing loaded model
        """
        try:
            import torch
            from moirai_forecast import MoiraiForecast
        except ImportError:
            raise ImportError(
                "moirai is required. Install with: pip install moirai-forecast"
            )

        # Load pretrained model
        self._model = MoiraiForecast(
            module_name=self.model_name,
            prediction_length=dataset.config.h,
            context_length=dataset.min_length,
        )

        return {
            "model": self._model,
            "model_name": self.model_name,
            "adapter": self,
        }

    def predict(
        self,
        dataset: TSDataset,
        artifact: dict[str, Any],
        h: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using Moirai.

        Args:
            dataset: Time-series dataset
            artifact: Model artifact from fit()
            h: Forecast horizon

        Returns:
            Forecast DataFrame with columns [unique_id, ds, yhat]
        """
        import numpy as np

        model = artifact["model"]
        horizon = h or dataset.config.h

        forecasts = []

        # Process each series
        for unique_id in dataset.df["unique_id"].unique():
            series_df = dataset.df[dataset.df["unique_id"] == unique_id].sort_values("ds")
            context = series_df["y"].values.astype(np.float32)

            # Generate forecast - Moirai returns distribution samples
            samples = model.predict(context, horizon)

            # Extract median forecast
            forecast_values = np.median(samples, axis=0)

            # Generate future timestamps
            last_date = series_df["ds"].iloc[-1]
            freq = dataset.config.freq
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                "unique_id": unique_id,
                "ds": future_dates,
                "yhat": forecast_values,
            })
            forecasts.append(forecast_df)

        return pd.concat(forecasts, ignore_index=True)


def fit(dataset: TSDataset, model_name: str = "moirai-1.1-R-small") -> dict[str, Any]:
    """Fit Moirai model (loads pretrained)."""
    adapter = MoiraiAdapter(model_name)
    return adapter.fit(dataset)


def predict(dataset: TSDataset, artifact: dict[str, Any], h: int | None = None) -> pd.DataFrame:
    """Generate forecasts from Moirai model."""
    adapter = artifact["adapter"]
    return adapter.predict(dataset, artifact, h)
