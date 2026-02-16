"""Tests for model fitting and prediction.

Tests statistical and TSFM model adapters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.models import fit, fit_tsfm, predict, predict_tsfm


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
        artifact = fit(dataset, "Naive")
        assert "sf" in artifact
        assert artifact["model_name"] == "Naive"

    def test_fit_seasonal_naive(self, sample_df, config):
        """Fit SeasonalNaive model."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "SeasonalNaive")
        assert "sf" in artifact
        assert artifact["model_name"] == "SeasonalNaive"

    def test_fit_historic_average(self, sample_df, config):
        """Fit HistoricAverage model."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "HistoricAverage")
        assert "sf" in artifact
        assert artifact["model_name"] == "HistoricAverage"

    def test_fit_unknown_model(self, sample_df, config):
        """Fit unknown model raises ValueError."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        with pytest.raises(ValueError, match="Unknown model"):
            fit(dataset, "UnknownModel")

    def test_fit_multi_series(self, multi_series_df, config):
        """Fit model with multiple series."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        artifact = fit(dataset, "Naive")
        assert "sf" in artifact


class TestStatisticalModelsPredict:
    """Test predict() function for statistical models."""

    def test_predict_basic(self, sample_df, config):
        """Basic prediction."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")
        forecast = predict(dataset, artifact, h=7)

        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

    def test_predict_default_h(self, sample_df, config):
        """Prediction uses config.h as default."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")
        forecast = predict(dataset, artifact)  # No h specified
        assert len(forecast) == config.h  # Should use config.h (7)

    def test_predict_multi_series(self, multi_series_df, config):
        """Predict with multiple series."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        artifact = fit(dataset, "Naive")
        forecast = predict(dataset, artifact, h=7)

        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

    def test_predict_seasonal_naive(self, sample_df, config):
        """Predict with SeasonalNaive."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "SeasonalNaive")
        forecast = predict(dataset, artifact, h=7)

        assert len(forecast) == 7
        assert "yhat" in forecast.columns

    def test_predict_historic_average(self, sample_df, config):
        """Predict with HistoricAverage."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "HistoricAverage")
        forecast = predict(dataset, artifact, h=7)

        assert len(forecast) == 7
        assert "yhat" in forecast.columns


class TestQuantilePrediction:
    """Test quantile prediction functionality."""

    def test_predict_with_quantiles(self, sample_df, config):
        """Predict with quantiles."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")
        forecast = predict(dataset, artifact, h=7, quantiles=[0.1, 0.5, 0.9])

        assert "q0.1" in forecast.columns
        assert "q0.5" in forecast.columns
        assert "q0.9" in forecast.columns

    def test_quantile_ordering(self, sample_df, config):
        """Quantiles should be ordered correctly."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")
        forecast = predict(dataset, artifact, h=7, quantiles=[0.1, 0.5, 0.9])

        # Lower quantile should be less than higher quantile
        assert all(forecast["q0.1"] <= forecast["q0.9"])
        assert all(forecast["q0.1"] <= forecast["q0.5"])
        assert all(forecast["q0.5"] <= forecast["q0.9"])

    def test_median_quantile_near_mean(self, sample_df, config):
        """Median quantile should be near mean (yhat)."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "HistoricAverage")
        forecast = predict(dataset, artifact, h=7, quantiles=[0.5])

        # Median should be close to yhat
        assert np.allclose(forecast["q0.5"], forecast["yhat"], rtol=0.1)


class TestTSFMAdapters:
    """Test TSFM adapter functions."""

    def test_fit_tsfm_unknown_adapter(self, sample_df, config):
        """Unknown TSFM adapter raises ValueError."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        with pytest.raises(ValueError, match="Unknown TSFM adapter"):
            fit_tsfm(dataset, "unknown_adapter")

    @pytest.mark.skip(reason="TSFMs are now default dependencies - import errors won't occur")
    def test_fit_tsfm_import_error(self, sample_df, config, monkeypatch):
        """TSFM adapter import error handling - DEPRECATED.

        TSFMs are now default dependencies, so this test is no longer valid.
        Import guards have been removed since chronos, tsagentkit-timesfm,
        and tsagentkit-uni2ts are required dependencies.
        """
        pass

    def test_predict_tsfm_invalid_artifact(self, sample_df, config):
        """Invalid TSFM artifact raises ValueError."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        invalid_artifact = {"not_adapter": "invalid"}

        with pytest.raises(ValueError, match="Invalid TSFM artifact"):
            predict_tsfm(dataset, invalid_artifact, h=7)

    def test_predict_tsfm_no_predict_method(self, sample_df, config):
        """TSFM artifact without predict method raises ValueError."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        invalid_artifact = {"adapter": object()}  # Object without predict method

        with pytest.raises(ValueError, match="Invalid TSFM artifact"):
            predict_tsfm(dataset, invalid_artifact, h=7)


class TestModelConsistency:
    """Test model consistency across different scenarios."""

    def test_same_data_same_forecast(self, sample_df, config):
        """Same data produces same forecast."""
        dataset = TSDataset.from_dataframe(sample_df, config)

        artifact1 = fit(dataset, "Naive")
        forecast1 = predict(dataset, artifact1, h=7)

        artifact2 = fit(dataset, "Naive")
        forecast2 = predict(dataset, artifact2, h=7)

        pd.testing.assert_series_equal(forecast1["yhat"], forecast2["yhat"])

    def test_forecast_increases_with_horizon(self, sample_df, config):
        """Forecast extends further with larger horizon."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")

        forecast_7 = predict(dataset, artifact, h=7)
        forecast_14 = predict(dataset, artifact, h=14)

        assert len(forecast_14) == 14
        assert len(forecast_7) == 7

    def test_forecast_dates_correct(self, sample_df, config):
        """Forecast dates are correct."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")

        forecast = predict(dataset, artifact, h=7)

        # First forecast date should be day after last data date
        last_data_date = sample_df["ds"].iloc[-1]
        first_forecast_date = forecast["ds"].iloc[0]
        expected_first_date = last_data_date + pd.Timedelta(days=1)
        assert first_forecast_date == expected_first_date

        # All dates should be consecutive
        date_diffs = forecast["ds"].diff().dropna()
        assert all(date_diffs == pd.Timedelta(days=1))


class TestForecastResult:
    """Test ForecastResult output structure."""

    def test_forecast_structure(self, sample_df, config):
        """Forecast has expected structure."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")
        forecast_df = predict(dataset, artifact, h=7)

        # Check all required columns exist
        required_cols = ["unique_id", "ds", "yhat"]
        for col in required_cols:
            assert col in forecast_df.columns

    def test_forecast_no_missing_values(self, sample_df, config):
        """Forecast has no missing values."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")
        forecast_df = predict(dataset, artifact, h=7)

        assert not forecast_df["yhat"].isnull().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
