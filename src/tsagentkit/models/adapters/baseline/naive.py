"""Naive baseline adapter.

Simple forecasting method: last value is the forecast for all horizons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def load() -> None:
    """Naive model is stateless, no loading needed."""
    return None


def fit(dataset: TSDataset) -> None:
    """Naive model is stateless, no fitting needed."""
    return None


def predict(artifact: None, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate naive forecast (last value repeated).

    Args:
        artifact: Unused (stateless model)
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    forecasts = []

    for unique_id in dataset.df["unique_id"].unique():
        # Get series data
        mask = dataset.df["unique_id"] == unique_id
        series_df = dataset.df[mask].sort_values("ds")

        # Last value is the forecast
        last_value = series_df["y"].iloc[-1]
        last_date = series_df["ds"].iloc[-1]

        # Generate future dates
        future_dates = pd.date_range(
            start=last_date,
            periods=h + 1,
            freq=dataset.config.freq,
        )[1:]

        # Create forecast
        forecast_df = pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": future_dates,
                "yhat": [last_value] * h,
            }
        )
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
