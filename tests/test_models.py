"""Tests for model fitting and prediction.

Tests statistical and TSFM model adapters using the protocol API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.models import fit, predict, get_spec


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    return pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50),
        "y": np.sin(np.linspace(0, 4 * np.pi, 50)) * 10 + 20,
    })


@pytest.fixture
def multi_series_df():
    """Create multi-series DataFrame."""
    np.random.seed(42)
    dfs = []
    for uid in ["A", "B"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 50,
            "ds": pd.date_range("2024-01-01", periods=50),
            "y": np.sin(np.linspace(0, 4 * np.pi, 50)) * 10 + 20 + np.random.randn(50) * 2,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def config():
    """Create default config."""
    return ForecastConfig(h=7, freq="D")


class TestStatisticalModelsFit:
    """Test fit() function for statistical models."""

    def test_fit_naive(self, sample_df, config):
        """Fit Naive model."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        assert artifact is None  # Naive is stateless

    def test_fit_seasonal_naive(self, sample_df, config):
        """Fit SeasonalNaive model."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("seasonal_naive")
        artifact = fit(spec, dataset)
        assert "season_length" in artifact

    def test_fit_unknown_model(self, sample_df, config):
        """Fit unknown model raises KeyError."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        with pytest.raises(KeyError, match="unknown_model"):
            spec = get_spec("unknown_model")
            fit(spec, dataset)

    def test_fit_multi_series(self, multi_series_df, config):
        """Fit model with multiple series."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        assert artifact is None


class TestStatisticalModelsPredict:
    """Test predict() function for statistical models."""

    def test_predict_basic(self, sample_df, config):
        """Basic prediction."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        forecast = predict(spec, artifact, dataset, h=7)

        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

    def test_predict_multi_series(self, multi_series_df, config):
        """Predict with multiple series."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        forecast = predict(spec, artifact, dataset, h=7)

        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

    def test_predict_seasonal_naive(self, sample_df, config):
        """Predict with SeasonalNaive."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("seasonal_naive")
        artifact = fit(spec, dataset)
        forecast = predict(spec, artifact, dataset, h=7)

        assert len(forecast) == 7
        assert "yhat" in forecast.columns

    def test_predict_naive_values(self, sample_df, config):
        """Naive forecast values are last observed values."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        forecast = predict(spec, artifact, dataset, h=7)

        last_value = sample_df["y"].iloc[-1]
        assert all(forecast["yhat"] == last_value)


class TestModelConsistency:
    """Test model consistency across different scenarios."""

    def test_same_data_same_forecast(self, sample_df, config):
        """Same data produces same forecast."""
        dataset = TSDataset.from_dataframe(sample_df, config)

        spec = get_spec("naive")
        artifact1 = fit(spec, dataset)
        forecast1 = predict(spec, artifact1, dataset, h=7)

        artifact2 = fit(spec, dataset)
        forecast2 = predict(spec, artifact2, dataset, h=7)

        pd.testing.assert_series_equal(forecast1["yhat"], forecast2["yhat"])

    def test_forecast_increases_with_horizon(self, sample_df, config):
        """Forecast extends further with larger horizon."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)

        forecast_7 = predict(spec, artifact, dataset, h=7)
        forecast_14 = predict(spec, artifact, dataset, h=14)

        assert len(forecast_14) == 14
        assert len(forecast_7) == 7

    def test_forecast_dates_correct(self, sample_df, config):
        """Forecast dates are correct."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)

        forecast = predict(spec, artifact, dataset, h=7)

        # First forecast date should be day after last data date
        last_data_date = sample_df["ds"].iloc[-1]
        first_forecast_date = forecast["ds"].iloc[0]
        expected_first_date = last_data_date + pd.Timedelta(days=1)
        assert first_forecast_date == expected_first_date

        # All dates should be consecutive
        date_diffs = forecast["ds"].diff().dropna()
        assert all(date_diffs == pd.Timedelta(days=1))


class TestForecastResult:
    """Test Forecast output structure."""

    def test_forecast_structure(self, sample_df, config):
        """Forecast has expected structure."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        forecast_df = predict(spec, artifact, dataset, h=7)

        # Check all required columns exist
        required_cols = ["unique_id", "ds", "yhat"]
        for col in required_cols:
            assert col in forecast_df.columns

    def test_forecast_no_missing_values(self, sample_df, config):
        """Forecast has no missing values."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        forecast_df = predict(spec, artifact, dataset, h=7)

        assert not forecast_df["yhat"].isnull().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
