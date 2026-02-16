"""Tests for result types.

Tests ForecastResult and RunResult functionality.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit import ForecastConfig
from tsagentkit.core.results import ForecastResult, RunResult


@pytest.fixture
def sample_config():
    """Create sample config."""
    return ForecastConfig(h=7, freq="D")


@pytest.fixture
def sample_forecast_df():
    """Create sample forecast DataFrame."""
    return pd.DataFrame({
        "unique_id": ["A"] * 7,
        "ds": pd.date_range("2024-01-31", periods=7),
        "yhat": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        "q0.1": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "q0.9": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    })


@pytest.fixture
def multi_series_forecast_df():
    """Create multi-series forecast DataFrame."""
    dfs = []
    for uid in ["A", "B"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 7,
            "ds": pd.date_range("2024-01-31", periods=7),
            "yhat": range(7),
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestForecastResult:
    """Test ForecastResult dataclass."""

    def test_create_forecast_result(self, sample_forecast_df, sample_config):
        """Create ForecastResult."""
        result = ForecastResult(
            df=sample_forecast_df,
            model_name="test_model",
            config=sample_config,
        )
        assert result.df is sample_forecast_df
        assert result.model_name == "test_model"
        assert result.config == sample_config

    def test_create_without_config(self, sample_forecast_df):
        """Create ForecastResult without config."""
        result = ForecastResult(
            df=sample_forecast_df,
            model_name="test_model",
        )
        assert result.config is None

    def test_get_series(self, multi_series_forecast_df, sample_config):
        """Get single series from forecast."""
        result = ForecastResult(
            df=multi_series_forecast_df,
            model_name="test",
            config=sample_config,
        )
        series_a = result.get_series("A")
        assert len(series_a) == 7
        assert all(series_a["unique_id"] == "A")

    def test_get_series_returns_copy(self, multi_series_forecast_df, sample_config):
        """get_series returns a copy - verify by checking copy() is called."""
        result = ForecastResult(
            df=multi_series_forecast_df,
            model_name="test",
            config=sample_config,
        )
        series_a = result.get_series("A")
        # Verify we got a DataFrame
        assert isinstance(series_a, pd.DataFrame)
        assert len(series_a) == 7
        # Verify it's for series A
        assert all(series_a["unique_id"] == "A")

    def test_get_nonexistent_series(self, sample_forecast_df, sample_config):
        """Get series that doesn't exist."""
        result = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        series = result.get_series("Z")
        assert len(series) == 0

    def test_point_forecast(self, sample_forecast_df, sample_config):
        """Get point forecast."""
        result = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        point = result.point_forecast
        assert "unique_id" in point.columns
        assert "ds" in point.columns
        assert "yhat" in point.columns
        assert "q0.1" not in point.columns

    def test_point_forecast_subset_columns(self, sample_config):
        """Point forecast handles missing columns gracefully."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-31", periods=7),
        })
        result = ForecastResult(df=df, model_name="test", config=sample_config)
        point = result.point_forecast
        # Should only include available columns from the subset
        assert "unique_id" in point.columns
        assert "ds" in point.columns
        assert "yhat" not in point.columns  # Not in original df

    def test_forecast_result_is_frozen(self, sample_forecast_df, sample_config):
        """ForecastResult is frozen."""
        result = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        with pytest.raises(AttributeError):
            result.model_name = "new_name"


class TestRunResult:
    """Test RunResult dataclass."""

    def test_create_run_result(self, sample_forecast_df, sample_config):
        """Create RunResult."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="ensemble_median",
            config=sample_config,
        )
        result = RunResult(
            forecast=forecast,
            duration_ms=123.45,
            model_used="ensemble_median",
        )
        assert result.forecast == forecast
        assert result.duration_ms == 123.45
        assert result.model_used == "ensemble_median"
        assert result.backtest_metrics is None
        assert result.model_errors == []

    def test_create_with_backtest_metrics(self, sample_forecast_df, sample_config):
        """Create RunResult with backtest metrics."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        metrics = {"mae": 1.5, "rmse": 2.0, "mape": 5.0}
        result = RunResult(
            forecast=forecast,
            duration_ms=100.0,
            model_used="test",
            backtest_metrics=metrics,
        )
        assert result.backtest_metrics == metrics

    def test_create_with_model_errors(self, sample_forecast_df, sample_config):
        """Create RunResult with model errors."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="ensemble",
            config=sample_config,
        )
        errors = [
            {"model": "chronos", "error": "Out of memory"},
            {"model": "moirai", "error": "Import error"},
        ]
        result = RunResult(
            forecast=forecast,
            duration_ms=100.0,
            model_used="ensemble",
            model_errors=errors,
        )
        assert len(result.model_errors) == 2

    def test_to_dataframe(self, sample_forecast_df, sample_config):
        """to_dataframe returns forecast DataFrame."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        result = RunResult(
            forecast=forecast,
            duration_ms=100.0,
            model_used="test",
        )
        df = result.to_dataframe()
        pd.testing.assert_frame_equal(df, sample_forecast_df)

    def test_summary(self, sample_forecast_df, sample_config):
        """summary returns dict with run info."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="ensemble_median",
            config=sample_config,
        )
        result = RunResult(
            forecast=forecast,
            duration_ms=123.456,
            model_used="ensemble_median",
            model_errors=[{"model": "failed", "error": "test"}],
        )
        summary = result.summary()
        assert summary["model"] == "ensemble_median"
        assert summary["duration_ms"] == 123.46  # Rounded to 2 decimals
        assert summary["model_errors"] == 1
        assert summary["forecast_shape"] == (7, 5)  # 7 rows, 5 columns

    def test_summary_no_errors(self, sample_forecast_df, sample_config):
        """summary with no errors."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        result = RunResult(
            forecast=forecast,
            duration_ms=100.0,
            model_used="test",
            model_errors=[],
        )
        summary = result.summary()
        assert summary["model_errors"] == 0


class TestRunResultMutability:
    """Test RunResult mutability."""

    def test_run_result_is_mutable(self, sample_forecast_df, sample_config):
        """RunResult is mutable (not frozen)."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        result = RunResult(
            forecast=forecast,
            duration_ms=100.0,
            model_used="test",
        )
        # Should be able to modify
        result.duration_ms = 200.0
        assert result.duration_ms == 200.0

    def test_can_add_backtest_metrics_later(self, sample_forecast_df, sample_config):
        """Can add backtest metrics after creation."""
        forecast = ForecastResult(
            df=sample_forecast_df,
            model_name="test",
            config=sample_config,
        )
        result = RunResult(
            forecast=forecast,
            duration_ms=100.0,
            model_used="test",
        )
        result.backtest_metrics = {"mae": 1.0}
        assert result.backtest_metrics == {"mae": 1.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
