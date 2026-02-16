"""Tests for v2.0 TSFM ensemble forecasting.

Tests the new ensemble approach where all models (TSFM + statistical)
participate in the final forecast via median/mean aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, forecast
from tsagentkit.core.data import TSDataset
from tsagentkit.pipeline.stages import _compute_ensemble
from tsagentkit.router import build_plan, inspect_tsfm_adapters


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    return pd.DataFrame({
        "unique_id": ["A"] * 30,
        "ds": pd.date_range("2024-01-01", periods=30),
        "y": range(30),
    })


@pytest.fixture
def multi_series_data():
    """Create multi-series data."""
    dfs = []
    for uid in ["A", "B", "C"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": np.random.randn(30).cumsum() + 10,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestEnsembleConfig:
    """Test ensemble configuration options."""

    def test_default_ensemble_method_is_median(self, sample_data):
        """Default ensemble method should be median."""
        config = ForecastConfig(h=7, freq="D", tsfm_mode="disabled")
        assert config.ensemble_method == "median"

    def test_ensemble_mean_configuration(self, sample_data):
        """Can configure mean ensemble method."""
        config = ForecastConfig(h=7, freq="D", ensemble_method="mean")
        assert config.ensemble_method == "mean"

    def test_require_all_tsfm_option(self):
        """Can set require_all_tsfm option."""
        config = ForecastConfig(h=7, freq="D", require_all_tsfm=True)
        assert config.require_all_tsfm is True

    def test_min_models_for_ensemble_validation(self):
        """Min models for ensemble must be at least 1."""
        config = ForecastConfig(h=7, freq="D", min_models_for_ensemble=1)
        assert config.min_models_for_ensemble == 1


class TestBuildPlan:
    """Test ensemble plan building."""

    def test_build_plan_with_tsfm_disabled(self, sample_data):
        """Plan with TSFM disabled should only have statistical models."""
        dataset = TSDataset.from_dataframe(
            sample_data,
            ForecastConfig(h=7, freq="D"),
        )
        plan = build_plan(dataset, tsfm_mode="disabled", allow_fallback=True)

        assert len(plan.tsfm_models) == 0
        assert len(plan.statistical_models) == 3  # SeasonalNaive, HistoricAverage, Naive
        assert len(plan.all_models()) == 3

    def test_build_plan_ensemble_configuration(self, sample_data):
        """Plan should include ensemble configuration."""
        dataset = TSDataset.from_dataframe(
            sample_data,
            ForecastConfig(h=7, freq="D"),
        )
        plan = build_plan(
            dataset,
            tsfm_mode="disabled",
            ensemble_method="mean",
            require_all_tsfm=True,
        )

        assert plan.ensemble_method == "mean"
        assert plan.require_all_tsfm is True


class TestComputeEnsemble:
    """Test ensemble computation functions."""

    def test_median_ensemble_single_prediction(self):
        """Median ensemble with single prediction returns that prediction."""
        pred = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-31", periods=7),
            "yhat": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        })

        result = _compute_ensemble([pred], "median")

        assert len(result) == 7
        assert list(result["yhat"]) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    def test_median_ensemble_multiple_predictions(self):
        """Median ensemble computes element-wise median."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [30.0, 20.0, 10.0],
        })
        pred3 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [20.0, 20.0, 20.0],
        })

        result = _compute_ensemble([pred1, pred2, pred3], "median")

        assert list(result["yhat"]) == [20.0, 20.0, 20.0]

    def test_mean_ensemble_multiple_predictions(self):
        """Mean ensemble computes element-wise mean."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [30.0, 20.0, 10.0],
        })

        result = _compute_ensemble([pred1, pred2], "mean")

        assert list(result["yhat"]) == [20.0, 20.0, 20.0]

    def test_ensemble_includes_count(self):
        """Ensemble result includes count of contributing models."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-31", periods=3),
            "yhat": [20.0, 30.0, 40.0],
        })

        result = _compute_ensemble([pred1, pred2], "median")

        assert "_ensemble_count" in result.columns
        assert all(result["_ensemble_count"] == 2)


class TestIntegration:
    """Integration tests for ensemble forecasting."""

    def test_forecast_returns_ensemble_result(self, sample_data):
        """Forecast should return ensemble result with statistical models."""
        result = forecast(
            sample_data,
            h=7,
            freq="D",
            tsfm_mode="disabled",
            ensemble_method="median",
        )

        assert result.model_used == "ensemble_median"
        assert len(result.forecast.df) == 7
        assert "_ensemble_count" in result.forecast.df.columns
        assert result.forecast.df["_ensemble_count"].iloc[0] == 3  # 3 statistical models

    def test_forecast_mean_ensemble(self, sample_data):
        """Forecast with mean ensemble method."""
        result = forecast(
            sample_data,
            h=7,
            freq="D",
            tsfm_mode="disabled",
            ensemble_method="mean",
        )

        assert result.model_used == "ensemble_mean"

    def test_multi_series_ensemble(self, multi_series_data):
        """Ensemble works with multiple series."""
        result = forecast(
            multi_series_data,
            h=7,
            freq="D",
            tsfm_mode="disabled",
        )

        # Should have 7 periods * 3 series = 21 rows
        assert len(result.forecast.df) == 21

    def test_ensemble_with_model_errors(self, sample_data):
        """Ensemble should track models that failed."""
        result = forecast(
            sample_data,
            h=7,
            freq="D",
            tsfm_mode="disabled",
        )

        # With only statistical models, all should succeed
        assert len(result.model_errors) == 0

    def test_forecast_summary(self, sample_data):
        """RunResult summary should work."""
        result = forecast(
            sample_data,
            h=7,
            freq="D",
            tsfm_mode="disabled",
        )

        summary = result.summary()
        assert summary["model"] == "ensemble_median"
        assert "duration_ms" in summary
        assert "forecast_shape" in summary
        assert summary["model_errors"] == 0


class TestInspectTSFMAdapters:
    """Test TSFM adapter inspection."""

    def test_inspect_returns_list(self):
        """inspect_tsfm_adapters should return a list."""
        adapters = inspect_tsfm_adapters()
        assert isinstance(adapters, list)

    def test_adapters_exist_in_codebase(self):
        """Adapters should exist in the codebase and be discoverable."""
        # The adapters are part of the codebase, so they should be listed
        adapters = inspect_tsfm_adapters()
        # The adapters (chronos, timesfm, moirai) exist in models/adapters/
        assert "chronos" in adapters
        assert "timesfm" in adapters
        assert "moirai" in adapters


class TestForecastConfigPresets:
    """Test ForecastConfig preset methods with ensemble."""

    def test_quick_preset(self):
        """Quick preset should work with ensemble."""
        config = ForecastConfig.quick(h=7, freq="D")
        assert config.ensemble_method == "median"
        assert config.n_backtest_windows == 2

    def test_standard_preset(self):
        """Standard preset should require TSFM by default."""
        config = ForecastConfig.standard(h=7, freq="D")
        assert config.tsfm_mode == "required"
        assert config.n_backtest_windows == 5

    def test_strict_preset(self):
        """Strict preset should require all TSFM."""
        config = ForecastConfig.strict(h=7, freq="D")
        assert config.tsfm_mode == "required"
        assert config.allow_fallback is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
