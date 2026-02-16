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

    for unique_id in dataset.df[dataset.config.id_col].unique():
        # Get series data
        mask = dataset.df[dataset.config.id_col] == unique_id
        series_df = dataset.df[mask].sort_values(dataset.config.time_col)

        # Last value is the forecast
        last_value = series_df[dataset.config.target_col].iloc[-1]
        last_date = series_df[dataset.config.time_col].iloc[-1]

        # Generate future dates
        future_dates = pd.date_range(
            start=last_date,
            periods=h + 1,
            freq=dataset.config.freq,
        )[1:]

        # Create forecast
        forecast_df = pd.DataFrame({
            dataset.config.id_col: unique_id,
            dataset.config.time_col: future_dates,
            "yhat": [last_value] * h,
        })
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
