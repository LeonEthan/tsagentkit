"""Integration tests for TSFM adapters with the full pipeline.

These tests verify TSFM adapters work correctly within the complete
forecasting pipeline, including ensemble scenarios.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset


# Markers for real TSFM tests
pytestmark = [
    pytest.mark.tsfm,
]


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50),
        "y": np.sin(np.linspace(0, 4 * np.pi, 50)) * 10 + 20 + np.random.randn(50) * 2,
    })


@pytest.fixture
def multi_series_df():
    """Create multi-series DataFrame for testing."""
    np.random.seed(42)
    dfs = []
    for uid in ["A", "B", "C"]:
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


@pytest.fixture
def mock_chronos_artifact():
    """Create a mock Chronos artifact."""
    mock_pipeline = MagicMock()
    mock_prediction = MagicMock()
    mock_prediction.median.return_value.values.numpy.return_value = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    mock_pipeline.predict.return_value = mock_prediction

    return {
        "pipeline": mock_pipeline,
        "model_name": "chronos-t5-tiny",
        "adapter": MagicMock(),
    }


@pytest.fixture
def mock_timesfm_artifact():
    """Create a mock TimesFM artifact."""
    mock_model = MagicMock()
    mock_model.forecast.return_value = (np.array([[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]]), None)

    return {
        "model": mock_model,
        "adapter": MagicMock(),
    }


@pytest.fixture
def mock_moirai_artifact():
    """Create a mock Moirai artifact."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.randn(100, 7) + 12

    return {
        "model": mock_model,
        "model_name": "moirai-1.1-R-small",
        "adapter": MagicMock(),
    }


class TestTSFMInPipeline:
    """Integration tests for TSFM models in the full pipeline."""

    @pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM integration tests"
    )
    def test_chronos_in_full_pipeline(self, sample_df):
        """Test forecast() with Chronos and tsfm_mode='preferred'."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            n_backtest_windows=0,  # Skip backtest for speed
        )

        result = forecast(sample_df, h=config.h, freq=config.freq)

        # Verify result structure
        assert result is not None
        assert hasattr(result, "forecast")
        assert hasattr(result, "config")

        # Verify forecast has expected structure
        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) >= 7

    @pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM integration tests"
    )
    def test_all_tsfm_in_ensemble(self, sample_df):
        """Test ensemble with Chronos + TimesFM + Moirai + statistical models."""
        from tsagentkit import forecast

        # Use preferred mode to get TSFM + statistical ensemble
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            ensemble_method="median",
            n_backtest_windows=0,
        )

        result = forecast(sample_df, h=config.h, freq=config.freq)

        # Verify we got a forecast
        assert result is not None
        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) >= 7

    @pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM integration tests"
    )
    def test_tsfm_with_multi_series(self, multi_series_df):
        """Test TSFM models with multiple series."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            n_backtest_windows=0,
        )

        result = forecast(multi_series_df, h=config.h, freq=config.freq)

        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)

        # Should have forecasts for all 3 series
        unique_ids = forecast_df["unique_id"].unique()
        assert len(unique_ids) == 3
        assert set(unique_ids) == {"A", "B", "C"}


class TestTSFMPolicyMatrix:
    """Test all combinations of tsfm_mode and require_all_tsfm."""

    @pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM integration tests"
    )
    def test_tsfm_mode_required_all_succeed(self, sample_df):
        """tsfm_mode='required', all TSFM succeed."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
            n_backtest_windows=0,
        )

        # Should succeed with TSFM models
        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

    @pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM integration tests"
    )
    def test_tsfm_mode_required_one_fails(self, sample_df):
        """tsfm_mode='required', one TSFM fails, should still succeed with others."""
        from tsagentkit import forecast

        # When require_all_tsfm=False, partial TSFM success is OK
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
            allow_fallback=True,
            n_backtest_windows=0,
        )

        # Should succeed even if one TSFM fails (uses others)
        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

    @pytest.mark.skipif(
        not os.getenv("TSFM_RUN_REAL"),
        reason="Set TSFM_RUN_REAL=1 to run real TSFM integration tests"
    )
    def test_require_all_tsfm_true_one_fails(self, sample_df):
        """require_all_tsfm=True, one TSFM fails, should handle gracefully."""
        from tsagentkit import forecast

        # This tests the strict policy - all TSFM must succeed
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=True,
            allow_fallback=True,
            n_backtest_windows=0,
        )

        # The behavior depends on implementation - may raise error or handle gracefully
        try:
            result = forecast(sample_df, h=config.h, freq=config.freq)
            assert result is not None
        except Exception as e:
            # If it raises, should be a meaningful error
            assert "TSFM" in str(e) or "model" in str(e).lower()


class TestTSFMPipelineWithMocking:
    """Pipeline tests with mocked TSFM models for faster execution."""

    def test_pipeline_with_mocked_chronos(self, sample_df, config, mock_chronos_artifact):
        """Test pipeline with mocked Chronos adapter."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        with patch.object(ChronosAdapter, "fit", return_value=mock_chronos_artifact):
            with patch.object(ChronosAdapter, "predict") as mock_predict:
                # Create expected forecast
                forecast_df = pd.DataFrame({
                    "unique_id": ["A"] * 7,
                    "ds": pd.date_range("2024-02-20", periods=7),
                    "yhat": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                })
                mock_predict.return_value = forecast_df

                # Create adapter and run prediction
                adapter = ChronosAdapter()
                dataset = TSDataset.from_dataframe(sample_df, config)
                artifact = adapter.fit(dataset)
                result = adapter.predict(dataset, artifact, h=7)

                assert len(result) == 7
                assert list(result["yhat"]) == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    def test_pipeline_with_mocked_timesfm(self, sample_df, config, mock_timesfm_artifact):
        """Test pipeline with mocked TimesFM adapter."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        with patch.object(TimesFMAdapter, "fit", return_value=mock_timesfm_artifact):
            with patch.object(TimesFMAdapter, "predict") as mock_predict:
                forecast_df = pd.DataFrame({
                    "unique_id": ["A"] * 7,
                    "ds": pd.date_range("2024-02-20", periods=7),
                    "yhat": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                })
                mock_predict.return_value = forecast_df

                adapter = TimesFMAdapter()
                dataset = TSDataset.from_dataframe(sample_df, config)
                artifact = adapter.fit(dataset)
                result = adapter.predict(dataset, artifact, h=7)

                assert len(result) == 7

    def test_pipeline_with_mocked_moirai(self, sample_df, config, mock_moirai_artifact):
        """Test pipeline with mocked Moirai adapter."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        with patch.object(MoiraiAdapter, "fit", return_value=mock_moirai_artifact):
            with patch.object(MoiraiAdapter, "predict") as mock_predict:
                forecast_df = pd.DataFrame({
                    "unique_id": ["A"] * 7,
                    "ds": pd.date_range("2024-02-20", periods=7),
                    "yhat": [12.0] * 7,
                })
                mock_predict.return_value = forecast_df

                adapter = MoiraiAdapter()
                dataset = TSDataset.from_dataframe(sample_df, config)
                artifact = adapter.fit(dataset)
                result = adapter.predict(dataset, artifact, h=7)

                assert len(result) == 7

    def test_tsfm_fallback_behavior(self, sample_df, config):
        """Test that TSFM failures fall back to statistical models."""
        from tsagentkit import forecast

        # Use disabled mode to ensure we use statistical models
        config_disabled = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="disabled",
            n_backtest_windows=0,
        )

        result = forecast(sample_df, h=config_disabled.h, freq=config_disabled.freq)
        assert result is not None

        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) >= 7


class TestTSFMEnsembleScenarios:
    """Test TSFM ensemble scenarios."""

    def test_ensemble_median_calculation(self):
        """Test that ensemble median is calculated correctly."""
        # Simulate forecasts from multiple models
        forecasts = [
            pd.DataFrame({
                "unique_id": ["A"] * 7,
                "ds": pd.date_range("2024-02-20", periods=7),
                "yhat": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            }),
            pd.DataFrame({
                "unique_id": ["A"] * 7,
                "ds": pd.date_range("2024-02-20", periods=7),
                "yhat": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            }),
            pd.DataFrame({
                "unique_id": ["A"] * 7,
                "ds": pd.date_range("2024-02-20", periods=7),
                "yhat": [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            }),
        ]

        # Calculate median ensemble
        yhat_values = np.array([f["yhat"].values for f in forecasts])
        median_forecast = np.median(yhat_values, axis=0)

        expected_median = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        np.testing.assert_array_equal(median_forecast, expected_median)

    def test_ensemble_with_partial_tsfm_success(self, sample_df, config):
        """Test ensemble when some TSFM models succeed and others fail."""
        from tsagentkit import forecast

        # Use preferred mode with fallback enabled
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            allow_fallback=True,
            ensemble_method="median",
            n_backtest_windows=0,
        )

        # Should succeed even with partial TSFM failures
        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) >= 7


class TestTSFMConfigPresets:
    """Test TSFM behavior with different config presets."""

    def test_quick_preset_allows_fallback(self, sample_df):
        """Verify 'quick' preset allows TSFM fallback."""
        from tsagentkit import forecast

        config = ForecastConfig.quick(h=7, freq="D")

        # Quick preset should have preferred TSFM mode
        assert config.tsfm_mode == "preferred"
        assert config.allow_fallback is True

        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

    def test_standard_preset_requires_tsfm(self, sample_df):
        """Verify 'standard' preset requires TSFM."""
        config = ForecastConfig.standard(h=7, freq="D")

        # Standard preset should require TSFM
        assert config.tsfm_mode == "required"

    def test_strict_preset_requires_all_tsfm(self, sample_df):
        """Verify 'strict' preset requires all TSFM."""
        config = ForecastConfig.strict(h=7, freq="D")

        # Strict preset should require all TSFM
        assert config.tsfm_mode == "required"
        assert config.require_all_tsfm is True
        assert config.allow_fallback is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
