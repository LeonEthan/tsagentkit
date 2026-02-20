"""Seasonal Naive baseline adapter.

Forecast: value from the same season in the previous period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def load() -> None:
    """SeasonalNaive model is stateless, no loading needed."""
    return None


def fit(dataset: TSDataset) -> dict:
    """Extract season length from dataset config.

    Args:
        dataset: Time-series dataset

    Returns:
        Dictionary with season_length
    """
    season_length = dataset.config.season_length or 1
    return {"season_length": season_length}


def predict(artifact: dict, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate seasonal naive forecast.

    Args:
        artifact: Contains season_length
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    season_length = artifact["season_length"]
    forecasts = []

    for unique_id in dataset.df["unique_id"].unique():
        # Get series data
        mask = dataset.df["unique_id"] == unique_id
        series_df = dataset.df[mask].sort_values("ds")
        values = series_df["y"].values

        # Generate forecasts by repeating historical seasonal values
        forecast_values = []
        for i in range(h):
            # Index: go back season_length steps for each horizon step
            hist_idx = len(values) - season_length + (i % season_length)
            if hist_idx >= 0:
                forecast_values.append(values[hist_idx])
            else:
                # Fallback to last value if not enough history
                forecast_values.append(values[-1])

        # Generate future dates
        last_date = series_df["ds"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date,
            periods=h + 1,
            freq=dataset.config.freq,
        )[1:]

        # Create forecast
        forecast_df = pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": future_dates[: len(forecast_values)],
                "yhat": forecast_values,
            }
        )
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
