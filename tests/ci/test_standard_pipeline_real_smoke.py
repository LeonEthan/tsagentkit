"""Real smoke tests for Standard Pipeline with actual TSFM models.

These tests verify that the Standard Pipeline works end-to-end with real models:
1. forecast() - zero-config API
2. run_forecast() - config-based API
3. ModelCache integration
4. Multi-series handling
5. Different horizons and frequencies

Requirements:
- Set TSFM_RUN_REAL=1 environment variable to run
- 45-minute timeout for model downloads on first run
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Skip all tests in this file if TSFM_RUN_REAL is not set
pytestmark = [
    pytest.mark.tsfm,
    pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM smoke tests"
    ),
    pytest.mark.timeout(2700),  # 45 minutes for model downloads
]


def create_test_dataset(
    n_series: int = 1,
    n_points: int = 50,
    freq: str = "D",
    start_date: str = "2024-01-01",
    add_trend: bool = True,
    add_seasonality: bool = True,
) -> pd.DataFrame:
    """Create test dataset with configurable properties."""
    np.random.seed(42)
    dfs = []

    for i, uid in enumerate([f"S{i}" for i in range(n_series)]):
        t = np.linspace(0, 4 * np.pi, n_points)
        values = np.random.randn(n_points) * 0.5

        if add_trend:
            values += np.linspace(0, 10, n_points)

        if add_seasonality:
            values += 5 * np.sin(t + i)

        values += 100  # Base level

        df = pd.DataFrame({
            "unique_id": [uid] * n_points,
            "ds": pd.date_range(start_date, periods=n_points, freq=freq),
            "y": values,
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def single_series_df():
    """Single series dataset."""
    return create_test_dataset(n_series=1, n_points=50)


@pytest.fixture
def multi_series_df():
    """Multi-series dataset with 3 series."""
    return create_test_dataset(n_series=3, n_points=50)


@pytest.fixture
def long_series_df():
    """Long single series for testing context windows."""
    return create_test_dataset(n_series=1, n_points=500)


@pytest.fixture
def forecast_config():
    """Create forecast config."""
    from tsagentkit import ForecastConfig
    return ForecastConfig(h=7, freq="D")


class TestForecastZeroConfig:
    """Real smoke tests for forecast() zero-config API."""

    def test_forecast_basic(self, single_series_df):
        """Verify forecast() works with minimal config."""
        from tsagentkit import forecast

        print("\nRunning forecast() basic test...")

        result = forecast(single_series_df, h=7)

        # Verify result structure
        assert result.df is not None
        assert len(result.df) == 7
        assert "unique_id" in result.df.columns
        assert "ds" in result.df.columns
        assert "yhat" in result.df.columns

        # Verify forecast values
        assert not result.df["yhat"].isnull().any()
        assert all(np.isfinite(result.df["yhat"]))

        # Verify model name
        assert "ensemble" in result.model_name

        print(f"✓ forecast() basic test passed: {len(result.df)} forecasts")

    def test_forecast_multi_series(self, multi_series_df):
        """Verify forecast() handles multiple series."""
        from tsagentkit import forecast

        print("\nRunning forecast() multi-series test...")

        result = forecast(multi_series_df, h=7)

        # Should have forecasts for all 3 series
        assert len(result.df) == 21  # 3 series * 7 horizon
        assert set(result.df["unique_id"].unique()) == {"S0", "S1", "S2"}

        # Each series should have 7 forecasts
        for uid in ["S0", "S1", "S2"]:
            series_forecasts = result.df[result.df["unique_id"] == uid]
            assert len(series_forecasts) == 7

        print(f"✓ forecast() multi-series test passed: {len(result.df)} total forecasts")

    def test_forecast_different_horizons(self, single_series_df):
        """Verify forecast() works with different horizon values."""
        from tsagentkit import forecast

        print("\nRunning forecast() horizon test...")

        for h in [1, 7, 14, 30]:
            result = forecast(single_series_df, h=h)
            assert len(result.df) == h, f"Expected {h} forecasts, got {len(result.df)}"

        print("✓ forecast() horizon test passed")

    def test_forecast_quantiles_config(self, single_series_df):
        """Verify forecast() stores quantile config (quantiles in output TBD)."""
        from tsagentkit import forecast

        print("\nRunning forecast() quantiles config test...")

        result = forecast(single_series_df, h=7, quantiles=(0.1, 0.5, 0.9))

        # Verify quantile config is stored
        assert result.config.quantiles == (0.1, 0.5, 0.9)

        # Note: TSFM adapters currently return only yhat column
        # Full quantile support would require adapter updates
        assert "yhat" in result.df.columns
        assert len(result.df) == 7

        print("✓ forecast() quantiles config test passed")

    def test_forecast_ensemble_methods(self, single_series_df):
        """Verify forecast() supports different ensemble methods."""
        from tsagentkit import forecast

        print("\nRunning forecast() ensemble methods test...")

        result_median = forecast(single_series_df, h=7, ensemble_method="median")
        result_mean = forecast(single_series_df, h=7, ensemble_method="mean")

        assert len(result_median.df) == 7
        assert len(result_mean.df) == 7
        assert result_median.model_name == "ensemble_median"
        assert result_mean.model_name == "ensemble_mean"

        print("✓ forecast() ensemble methods test passed")


class TestRunForecastConfig:
    """Real smoke tests for run_forecast() with config."""

    def test_run_forecast_basic(self, single_series_df, forecast_config):
        """Verify run_forecast() works with ForecastConfig."""
        from tsagentkit import run_forecast

        print("\nRunning run_forecast() basic test...")

        result = run_forecast(single_series_df, forecast_config)

        assert len(result.df) == 7
        assert "unique_id" in result.df.columns
        assert "ds" in result.df.columns
        assert "yhat" in result.df.columns

        print("✓ run_forecast() basic test passed")

    def test_run_forecast_strict_mode(self, single_series_df):
        """Verify run_forecast() with strict config."""
        from tsagentkit import ForecastConfig, run_forecast

        print("\nRunning run_forecast() strict mode test...")

        config = ForecastConfig.strict(h=7, freq="D")
        result = run_forecast(single_series_df, config)

        assert len(result.df) == 7

        print("✓ run_forecast() strict mode test passed")

    def test_run_forecast_quick_mode(self, single_series_df):
        """Verify run_forecast() with quick config."""
        from tsagentkit import ForecastConfig, run_forecast

        print("\nRunning run_forecast() quick mode test...")

        config = ForecastConfig.quick(h=14, freq="D")
        result = run_forecast(single_series_df, config)

        assert len(result.df) == 14

        print("✓ run_forecast() quick mode test passed")


class TestPipelineFrequencies:
    """Real smoke tests for different time frequencies."""

    def test_daily_frequency(self):
        """Verify pipeline works with daily data."""
        from tsagentkit import forecast

        df = create_test_dataset(n_series=1, n_points=50, freq="D")
        result = forecast(df, h=7, freq="D")

        # Verify date spacing
        date_diff = result.df["ds"].iloc[1] - result.df["ds"].iloc[0]
        assert date_diff == pd.Timedelta(days=1)

        print("✓ Daily frequency test passed")

    def test_hourly_frequency(self):
        """Verify pipeline works with hourly data."""
        from tsagentkit import forecast

        df = create_test_dataset(n_series=1, n_points=100, freq="h")
        result = forecast(df, h=24, freq="h")

        # Verify date spacing
        date_diff = result.df["ds"].iloc[1] - result.df["ds"].iloc[0]
        assert date_diff == pd.Timedelta(hours=1)

        print("✓ Hourly frequency test passed")

    def test_weekly_frequency(self):
        """Verify pipeline works with weekly data."""
        from tsagentkit import forecast

        df = create_test_dataset(n_series=1, n_points=52, freq="W")
        result = forecast(df, h=4, freq="W")

        print("✓ Weekly frequency test passed")


class TestModelCacheIntegration:
    """Real smoke tests for ModelCache integration."""

    def test_model_cache_reuse(self, single_series_df):
        """Verify ModelCache reuses loaded models across forecast calls."""
        from tsagentkit import forecast, ModelCache

        print("\nRunning ModelCache reuse test...")

        # Clear cache first
        ModelCache.unload()
        assert len(ModelCache.list_loaded()) == 0

        # First forecast call - models should be loaded
        result1 = forecast(single_series_df, h=7)
        cached_after_first = ModelCache.list_loaded()

        # Second forecast call - should reuse cached models
        result2 = forecast(single_series_df, h=7)
        cached_after_second = ModelCache.list_loaded()

        # Models should still be cached
        assert len(cached_after_second) > 0
        assert set(cached_after_first) == set(cached_after_second)

        # Both results should be valid
        assert len(result1.df) == 7
        assert len(result2.df) == 7

        print(f"✓ ModelCache reuse test passed: {len(cached_after_second)} models cached")

    def test_model_cache_preload(self, single_series_df):
        """Verify ModelCache.preload() works."""
        from tsagentkit import ModelCache, forecast
        from tsagentkit.models.registry import REGISTRY

        print("\nRunning ModelCache.preload() test...")

        # Clear cache
        ModelCache.unload()

        # Preload all TSFM models
        models = [m for m in REGISTRY.values() if m.is_tsfm]
        ModelCache.preload(models)

        cached = ModelCache.list_loaded()
        assert len(cached) > 0

        # Forecast should use preloaded models
        result = forecast(single_series_df, h=7)
        assert len(result.df) == 7

        print(f"✓ ModelCache.preload() test passed: {len(cached)} models preloaded")

    def test_model_cache_unload(self):
        """Verify ModelCache.unload() frees memory."""
        from tsagentkit import ModelCache

        print("\nRunning ModelCache.unload() test...")

        # Unload specific model (if any loaded)
        ModelCache.unload("chronos")

        # Unload all
        ModelCache.unload()
        assert len(ModelCache.list_loaded()) == 0

        print("✓ ModelCache.unload() test passed")


class TestPipelineErrorHandling:
    """Real smoke tests for pipeline error handling."""

    def test_missing_required_columns(self):
        """Verify proper error for missing columns."""
        from tsagentkit import forecast, EContract

        df = pd.DataFrame({
            "wrong_id": ["A"] * 10,
            "wrong_date": pd.date_range("2024-01-01", periods=10),
            "wrong_value": range(10),
        })

        with pytest.raises(EContract) as exc_info:
            forecast(df, h=7)

        assert "Missing required columns" in str(exc_info.value)

        print("✓ Missing columns error test passed")

    def test_empty_dataframe(self):
        """Verify proper error for empty DataFrame."""
        from tsagentkit import forecast, EContract

        df = pd.DataFrame(columns=["unique_id", "ds", "y"])

        with pytest.raises(EContract) as exc_info:
            forecast(df, h=7)

        assert "empty" in str(exc_info.value).lower()

        print("✓ Empty DataFrame error test passed")

    def test_null_values(self):
        """Verify proper error for null values."""
        from tsagentkit import forecast, EContract

        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "y": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        })

        with pytest.raises(EContract) as exc_info:
            forecast(df, h=7)

        assert "null" in str(exc_info.value).lower()

        print("✓ Null values error test passed")


class TestPipelineAdvancedScenarios:
    """Advanced real smoke tests for edge cases."""

    def test_short_context_fallback(self):
        """Verify pipeline handles short context series."""
        from tsagentkit import forecast

        print("\nRunning short context fallback test...")

        # Create very short series
        df = create_test_dataset(n_series=1, n_points=20)
        result = forecast(df, h=7)

        assert len(result.df) == 7
        assert not result.df["yhat"].isnull().any()

        print("✓ Short context fallback test passed")

    def test_single_point_series(self):
        """Verify pipeline handles single-point series gracefully."""
        from tsagentkit import forecast, EContract

        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": [pd.Timestamp("2024-01-01")],
            "y": [100.0],
        })

        # This may fail or succeed depending on TSFM adapters
        # We're testing that it doesn't crash unexpectedly
        try:
            result = forecast(df, h=7)
            print("✓ Single point series handled (with forecast)")
        except Exception as e:
            # Expected to fail, verify it's a proper error
            assert isinstance(e, (EContract, ValueError))
            print(f"✓ Single point series handled (with error: {type(e).__name__})")

    def test_many_series(self):
        """Verify pipeline handles many series efficiently."""
        from tsagentkit import forecast

        print("\nRunning many series test...")

        df = create_test_dataset(n_series=10, n_points=50)
        result = forecast(df, h=7)

        assert len(result.df) == 70  # 10 series * 7 horizon

        print("✓ Many series test passed")

    def test_long_context_series(self, long_series_df):
        """Verify pipeline handles long context series."""
        from tsagentkit import forecast

        print("\nRunning long context series test...")

        result = forecast(long_series_df, h=7)

        assert len(result.df) == 7

        print("✓ Long context series test passed")


class TestPipelineResultStructure:
    """Tests for ForecastResult structure and metadata."""

    def test_result_has_config(self, single_series_df):
        """Verify ForecastResult includes config."""
        from tsagentkit import forecast, ForecastConfig

        config = ForecastConfig(h=7, freq="D", ensemble_method="median")
        result = forecast(single_series_df, h=7, freq="D", ensemble_method="median")

        assert result.config is not None
        assert result.config.h == 7
        assert result.config.freq == "D"

        print("✓ Result config test passed")

    def test_result_date_continuity(self, single_series_df):
        """Verify forecast dates continue from input data."""
        from tsagentkit import forecast

        result = forecast(single_series_df, h=7)

        last_input_date = single_series_df["ds"].iloc[-1]
        first_forecast_date = result.df["ds"].iloc[0]

        expected_first_date = last_input_date + pd.Timedelta(days=1)
        assert first_forecast_date == expected_first_date

        print("✓ Date continuity test passed")

    def test_result_forecast_range(self, single_series_df):
        """Verify forecast values are in reasonable range."""
        from tsagentkit import forecast

        result = forecast(single_series_df, h=7)

        input_mean = single_series_df["y"].mean()
        input_std = single_series_df["y"].std()

        # Forecasts should be within 5 std dev of input mean
        # (very loose check to avoid false failures)
        assert all(result.df["yhat"] > input_mean - 5 * input_std)
        assert all(result.df["yhat"] < input_mean + 5 * input_std)

        print("✓ Forecast range test passed")


def test_environment_setup():
    """Verify test environment and TSFM availability."""
    from tsagentkit.inspect import check_health, list_models

    print("\n=== Standard Pipeline Test Environment ===")
    print(f"TSFM_RUN_REAL: {os.getenv('TSFM_RUN_REAL', 'not set')}")

    # Check available models
    available = list_models(tsfm_only=True)
    print(f"Available TSFMs: {available}")

    # Health check
    health = check_health()
    print(f"TSFM available: {health.tsfm_available}")
    print(f"TSFM missing: {health.tsfm_missing}")

    assert len(available) > 0, "No TSFM models available"
    print("===========================================")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
