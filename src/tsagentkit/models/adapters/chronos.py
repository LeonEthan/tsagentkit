"""Chronos2 TSFM adapter for tsagentkit.

Wraps Amazon's Chronos model for zero-shot time series forecasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


class ChronosAdapter:
    """Chronos2 adapter for zero-shot forecasting."""

    def __init__(self, model_name: str = "chronos-t5-small"):
        """Initialize adapter with model specification.

        Args:
            model_name: Chronos model size (chronos-t5-tiny/small/base/large)
        """
        self.model_name = model_name
        self._model = None
        self._pipeline = None

    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load pretrained Chronos model.

        Chronos is a zero-shot model, so fit() just loads the model.

        Args:
            dataset: Time-series dataset (used for validation only)

        Returns:
            Model artifact containing loaded model
        """
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos is required. Install with: pip install chronos-forecasting"
            )

        # Load pretrained model
        self._pipeline = ChronosPipeline.from_pretrained(
            f"amazon/{self.model_name}",
            device_map="auto",
        )

        return {
            "pipeline": self._pipeline,
            "model_name": self.model_name,
            "adapter": self,
        }

    def predict(
        self,
        dataset: TSDataset,
        artifact: dict[str, Any],
        h: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using Chronos.

        Args:
            dataset: Time-series dataset
            artifact: Model artifact from fit()
            h: Forecast horizon

        Returns:
            Forecast DataFrame with columns [unique_id, ds, yhat]
        """
        import torch

        pipeline = artifact["pipeline"]
        horizon = h or dataset.config.h

        forecasts = []

        # Process each series
        for unique_id in dataset.df["unique_id"].unique():
            series_df = dataset.df[dataset.df["unique_id"] == unique_id].sort_values("ds")
            context = series_df["y"].values

            # Convert to tensor
            context_tensor = torch.tensor(context, dtype=torch.float32)

            # Generate forecast
            with torch.no_grad():
                prediction = pipeline.predict(context_tensor, horizon)

            # Extract median forecast
            forecast_values = prediction.median(axis=1).values.numpy()

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


def fit(dataset: TSDataset, model_name: str = "chronos-t5-small") -> dict[str, Any]:
    """Fit Chronos model (loads pretrained)."""
    adapter = ChronosAdapter(model_name)
    return adapter.fit(dataset)


def predict(dataset: TSDataset, artifact: dict[str, Any], h: int | None = None) -> pd.DataFrame:
    """Generate forecasts from Chronos model."""
    adapter = artifact["adapter"]
    return adapter.predict(dataset, artifact, h)
