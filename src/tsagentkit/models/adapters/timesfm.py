"""TimesFM 2.5 TSFM adapter for tsagentkit.

Wraps Google's TimesFM model for zero-shot time series forecasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


class TimesFMAdapter:
    """TimesFM 2.5 adapter for zero-shot forecasting."""

    def __init__(self, context_len: int = 512, horizon_len: int = 128):
        """Initialize adapter with model specification.

        Args:
            context_len: Maximum context length for the model
            horizon_len: Maximum horizon length for the model
        """
        self.context_len = context_len
        self.horizon_len = horizon_len
        self._model = None

    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load pretrained TimesFM model.

        TimesFM is a zero-shot model, so fit() just loads the model.

        Args:
            dataset: Time-series dataset (used for validation only)

        Returns:
            Model artifact containing loaded model
        """
        try:
            from timesfm import TimesFm
        except ImportError:
            raise ImportError(
                "timesfm is required. Install with: pip install timesfm"
            )

        # Load pretrained model
        self._model = TimesFm(
            context_len=self.context_len,
            horizon_len=self.horizon_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",  # Use 'gpu' if available
        )

        # Load checkpoint
        self._model.load_from_checkpoint(repo_id="google/timesfm-2.0-500m-pytorch")

        return {
            "model": self._model,
            "adapter": self,
        }

    def predict(
        self,
        dataset: TSDataset,
        artifact: dict[str, Any],
        h: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using TimesFM.

        Args:
            dataset: Time-series dataset
            artifact: Model artifact from fit()
            h: Forecast horizon

        Returns:
            Forecast DataFrame with columns [unique_id, ds, yhat]
        """
        model = artifact["model"]
        horizon = h or dataset.config.h

        forecasts = []
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
        for unique_id in dataset.df["unique_id"].unique():
            series_df = dataset.df[dataset.df["unique_id"] == unique_id].sort_values("ds")
            context = series_df["y"].values.tolist()

            # Generate forecast
            forecast_values, _ = model.forecast(
                inputs=[context],
                freq=[tfm_freq],
            )

            # Extract forecast for requested horizon
            forecast_values = forecast_values[0][:horizon]

            # Generate future timestamps
            last_date = series_df["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                "unique_id": unique_id,
                "ds": future_dates[:len(forecast_values)],
                "yhat": forecast_values,
            })
            forecasts.append(forecast_df)

        return pd.concat(forecasts, ignore_index=True)


def fit(dataset: TSDataset) -> dict[str, Any]:
    """Fit TimesFM model (loads pretrained)."""
    adapter = TimesFMAdapter()
    return adapter.fit(dataset)


def predict(dataset: TSDataset, artifact: dict[str, Any], h: int | None = None) -> pd.DataFrame:
    """Generate forecasts from TimesFM model."""
    adapter = artifact["adapter"]
    return adapter.predict(dataset, artifact, h)


# Alias for consistency with naming convention
TimesfmAdapter = TimesFMAdapter
