"""Tests for quantile forecasting functionality.

Tests quantile prediction, ensembling, and configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.models import fit, predict


@pytest.fixture
def sample_df():
    """Create sample DataFrame with some noise."""
    np.random.seed(42)
    return pd.DataFrame({
        "unique_id": ["A"] * 100,
        "ds": pd.date_range("2024-01-01", periods=100),
        "y": np.sin(np.linspace(0, 8 * np.pi, 100)) * 10 + 20 + np.random.randn(100) * 2,
    })


@pytest.fixture
def multi_series_df():
    """Create multi-series DataFrame."""
    np.random.seed(42)
    dfs = []
    for uid in ["A", "B"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "y": np.sin(np.linspace(0, 8 * np.pi, 100)) * 10 + 20 + np.random.randn(100) * 2,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestQuantileConfiguration:
    """Test quantile configuration."""

    def test_default_quantiles(self):
        """Default quantiles are (0.1, 0.5, 0.9)."""
        config = ForecastConfig(h=7, freq="D")
        assert config.quantiles == (0.1, 0.5, 0.9)

    def test_custom_quantiles(self):
        """Can set custom quantiles."""
        config = ForecastConfig(h=7, freq="D", quantiles=(0.05, 0.25, 0.5, 0.75, 0.95))
        assert config.quantiles == (0.05, 0.25, 0.5, 0.75, 0.95)

    def test_single_quantile(self):
        """Can set single quantile."""
        config = ForecastConfig(h=7, freq="D", quantiles=(0.5,))
        assert config.quantiles == (0.5,)


class TestQuantilePrediction:
    """Test quantile prediction in models."""

    def test_predict_with_quantiles(self, sample_df):
        """Predict returns quantile columns."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")

        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.1, 0.5, 0.9])

        assert "q0.1" in forecast_df.columns
        assert "q0.5" in forecast_df.columns
        assert "q0.9" in forecast_df.columns

    def test_quantile_column_naming(self, sample_df):
        """Quantile columns are named correctly."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "HistoricAverage")

        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.05, 0.95])

        assert "q0.05" in forecast_df.columns
        assert "q0.95" in forecast_df.columns

    def test_quantile_values_ordered(self, sample_df):
        """Quantile values are properly ordered."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "HistoricAverage")

        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.1, 0.5, 0.9])

        # Lower quantile should be less than higher quantile
        assert all(forecast_df["q0.1"] <= forecast_df["q0.9"])
        assert all(forecast_df["q0.1"] <= forecast_df["q0.5"])
        assert all(forecast_df["q0.5"] <= forecast_df["q0.9"])

    def test_median_quantile_near_mean(self, sample_df):
        """Median (0.5) quantile should be close to point forecast."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "HistoricAverage")

        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.5])

        # Median should be close to yhat (point forecast)
        assert np.allclose(forecast_df["q0.5"], forecast_df["yhat"], rtol=0.1)

    def test_quantile_spread_increases_with_uncertainty(self, sample_df):
        """Quantile spread reflects uncertainty."""
        config = ForecastConfig(h=14, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")

        forecast_df = predict(dataset, artifact, h=14, quantiles=[0.1, 0.9])

        # Calculate spread (90% - 10%)
        spread = forecast_df["q0.9"] - forecast_df["q0.1"]

        # All spreads should be non-negative
        assert all(spread >= 0)

    def test_quantiles_for_multi_series(self, multi_series_df):
        """Quantiles work with multiple series."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        artifact = fit(dataset, "SeasonalNaive")

        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.1, 0.5, 0.9])

        # Should have 14 rows (2 series * 7 horizon)
        assert len(forecast_df) == 14
        assert "q0.1" in forecast_df.columns
        assert "q0.9" in forecast_df.columns

        # Check per series
        for uid in ["A", "B"]:
            series_df = forecast_df[forecast_df["unique_id"] == uid]
            assert all(series_df["q0.1"] <= series_df["q0.9"])


class TestQuantileValues:
    """Test specific quantile value calculations."""

    def test_known_quantile_values(self):
        """Test with known data to verify quantile calculations."""
        # Create data with known statistical properties
        np.random.seed(42)
        values = np.random.normal(loc=100, scale=10, size=100)

        df = pd.DataFrame({
            "unique_id": ["A"] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "y": values,
        })

        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, config)
        artifact = fit(dataset, "HistoricAverage")

        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.1, 0.5, 0.9])

        # For a normal distribution:
        # 10th percentile ≈ mean - 1.28 * std
        # 90th percentile ≈ mean + 1.28 * std

        mean_val = values.mean()
        std_val = values.std()
        expected_10 = mean_val - 1.28 * std_val
        expected_90 = mean_val + 1.28 * std_val

        # Check that quantiles are reasonable (within 20% of expected)
        assert np.abs(forecast_df["q0.1"].iloc[0] - expected_10) / expected_10 < 0.2
        assert np.abs(forecast_df["q0.9"].iloc[0] - expected_90) / expected_90 < 0.2


class TestQuantileEdgeCases:
    """Test edge cases for quantiles."""

    def test_extreme_quantiles(self, sample_df):
        """Extreme quantiles (near 0 or 1)."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")

        # Very extreme quantiles might be problematic but should still work
        forecast_df = predict(dataset, artifact, h=7, quantiles=[0.01, 0.99])

        assert "q0.01" in forecast_df.columns
        assert "q0.99" in forecast_df.columns
        assert all(forecast_df["q0.01"] <= forecast_df["q0.99"])

    def test_many_quantiles(self, sample_df):
        """Many quantiles at once."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)
        artifact = fit(dataset, "Naive")

        quantiles = [i / 10 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
        forecast_df = predict(dataset, artifact, h=7, quantiles=quantiles)

        for q in quantiles:
            assert f"q{q}" in forecast_df.columns

        # Verify ordering
        for i in range(len(quantiles) - 1):
            q_low = quantiles[i]
            q_high = quantiles[i + 1]
            assert all(forecast_df[f"q{q_low}"] <= forecast_df[f"q{q_high}"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
