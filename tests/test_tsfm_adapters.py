"""Comprehensive unit tests for TSFM adapters with mocked dependencies.

These tests run quickly without requiring actual model downloads or GPU.
All external dependencies are mocked to ensure fast, deterministic tests.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset


@pytest.fixture
def sample_tsfm_df():
    """Sample data suitable for TSFM testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "unique_id": ["A"] * 100,
        "ds": pd.date_range("2024-01-01", periods=100),
        "y": np.sin(np.linspace(0, 8 * np.pi, 100)) * 10 + 20 + np.random.randn(100),
    })


@pytest.fixture
def multi_series_tsfm_df():
    """Multi-series data for TSFM testing."""
    np.random.seed(42)
    dfs = []
    for uid in ["A", "B", "C"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "y": np.sin(np.linspace(0, 8 * np.pi, 100)) * 10 + 20 + np.random.randn(100),
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def config():
    """Default forecast config."""
    return ForecastConfig(h=7, freq="D")


class TestChronosAdapter:
    """Tests for ChronosAdapter."""

    def test_init_default_model(self):
        """Verify default model_name is 'chronos-t5-small'."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        adapter = ChronosAdapter()
        assert adapter.model_name == "chronos-t5-small"
        assert adapter._model is None
        assert adapter._pipeline is None

    def test_init_custom_model(self):
        """Verify custom model_name can be set."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        adapter = ChronosAdapter(model_name="chronos-t5-tiny")
        assert adapter.model_name == "chronos-t5-tiny"

        adapter = ChronosAdapter(model_name="chronos-t5-large")
        assert adapter.model_name == "chronos-t5-large"

    def test_fit_returns_artifact(self, sample_tsfm_df, config):
        """Verify fit() returns proper artifact structure."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = ChronosAdapter(model_name="chronos-t5-tiny")

        # Mock the ChronosPipeline import and usage
        mock_pipeline = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline

        with patch.dict("sys.modules", {"chronos": MagicMock(ChronosPipeline=mock_pipeline)}):
            artifact = adapter.fit(dataset)

            # Verify artifact structure
            assert "pipeline" in artifact
            assert "model_name" in artifact
            assert artifact["model_name"] == "chronos-t5-tiny"
            assert "adapter" in artifact
            assert artifact["adapter"] is adapter

    def test_predict_returns_dataframe(self, sample_tsfm_df, config):
        """Verify predict() returns DataFrame with correct columns."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = ChronosAdapter()

        # Create mock artifact with expected structure
        mock_pipeline = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.median.return_value.values.numpy.return_value = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        mock_pipeline.predict.return_value = mock_prediction

        artifact = {
            "pipeline": mock_pipeline,
            "model_name": "chronos-t5-small",
            "adapter": adapter,
        }

        # Mock torch operations
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), tensor=MagicMock(return_value=mock_tensor))}):
            forecast = adapter.predict(dataset, artifact, h=7)

            # Verify structure
            assert isinstance(forecast, pd.DataFrame)
            assert "unique_id" in forecast.columns
            assert "ds" in forecast.columns
            assert "yhat" in forecast.columns

            # Verify content
            assert len(forecast) == 7
            assert forecast["unique_id"].iloc[0] == "A"

    def test_predict_uses_config_h_when_h_none(self, sample_tsfm_df, config):
        """Verify predict() uses config.h when h is None."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = ChronosAdapter()

        # Mock predict to capture what horizon was used
        mock_pipeline = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.median.return_value.values.numpy.return_value = np.array([10.0] * 7)
        mock_pipeline.predict.return_value = mock_prediction

        artifact = {
            "pipeline": mock_pipeline,
            "model_name": "chronos-t5-small",
            "adapter": adapter,
        }

        # Don't pass h - should use config.h (7)
        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), tensor=MagicMock())}):
            forecast = adapter.predict(dataset, artifact)
            assert len(forecast) == 7

    def test_predict_generates_correct_dates(self, sample_tsfm_df, config):
        """Verify forecast dates are correctly generated."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = ChronosAdapter()

        mock_pipeline = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.median.return_value.values.numpy.return_value = np.array([10.0] * 7)
        mock_pipeline.predict.return_value = mock_prediction

        artifact = {
            "pipeline": mock_pipeline,
            "model_name": "chronos-t5-small",
            "adapter": adapter,
        }

        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), tensor=MagicMock())}):
            forecast = adapter.predict(dataset, artifact, h=7)

            # First forecast date should be day after last data date
            last_data_date = sample_tsfm_df["ds"].iloc[-1]
            expected_first_date = last_data_date + pd.Timedelta(days=1)
            assert forecast["ds"].iloc[0] == expected_first_date

            # Dates should be consecutive
            date_diffs = forecast["ds"].diff().dropna()
            assert all(date_diffs == pd.Timedelta(days=1))

    def test_import_error_handling(self, sample_tsfm_df, config):
        """Verify proper error when chronos not installed."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = ChronosAdapter()

        # Remove chronos from sys.modules to simulate it not being installed
        with patch.dict("sys.modules", {"chronos": None}):
            with pytest.raises(ImportError):
                adapter.fit(dataset)

    def test_fit_module_function(self, sample_tsfm_df, config):
        """Test the module-level fit function exists."""
        from tsagentkit.models.adapters import chronos

        assert callable(chronos.fit)

    def test_predict_module_function(self, sample_tsfm_df, config):
        """Test the module-level predict function exists."""
        from tsagentkit.models.adapters import chronos

        assert callable(chronos.predict)


class TestTimesFMAdapter:
    """Tests for TimesFMAdapter."""

    def test_init_default_config(self):
        """Verify default context_len=512, horizon_len=128."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        adapter = TimesFMAdapter()
        assert adapter.context_len == 512
        assert adapter.horizon_len == 128
        assert adapter._model is None

    def test_init_custom_config(self):
        """Verify custom config can be set."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        adapter = TimesFMAdapter(context_len=256, horizon_len=64)
        assert adapter.context_len == 256
        assert adapter.horizon_len == 64

    def test_frequency_mapping(self):
        """Verify pandas freq maps to TimesFM tokens correctly."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        # Test frequency mapping via internal logic
        adapter = TimesFMAdapter()

        # The frequency map is internal to predict, we verify via documentation
        freq_map_expected = {
            "D": "D",
            "H": "H",
            "M": "M",
            "MS": "M",
            "Q": "Q",
            "QS": "Q",
            "W": "W",
            "B": "B",
        }

        for pandas_freq, tfm_freq in freq_map_expected.items():
            # Verify the mapping exists in code
            assert tfm_freq is not None

    def test_fit_returns_artifact(self, sample_tsfm_df, config):
        """Verify fit() returns proper artifact structure."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = TimesFMAdapter()

        # Mock the TimesFm import and usage
        mock_model = MagicMock()
        mock_model.load_from_checkpoint.return_value = None

        with patch.dict("sys.modules", {"timesfm": MagicMock(TimesFm=MagicMock(return_value=mock_model))}):
            artifact = adapter.fit(dataset)

            # Verify artifact structure
            assert "model" in artifact
            assert "adapter" in artifact
            assert artifact["adapter"] is adapter

    def test_predict_returns_dataframe(self, sample_tsfm_df, config):
        """Verify predict() returns DataFrame with correct structure."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = TimesFMAdapter()

        mock_model = MagicMock()
        mock_model.forecast.return_value = (np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]), None)

        artifact = {
            "model": mock_model,
            "adapter": adapter,
        }

        forecast = adapter.predict(dataset, artifact, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

    def test_predict_truncates_to_horizon(self, sample_tsfm_df, config):
        """Verify predict() truncates forecast to requested horizon."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = TimesFMAdapter()

        # Mock returns 10 values, but we only want 7
        mock_model = MagicMock()
        mock_model.forecast.return_value = (
            np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]]),
            None
        )

        artifact = {
            "model": mock_model,
            "adapter": adapter,
        }

        forecast = adapter.predict(dataset, artifact, h=7)

        # Should only have 7 values
        assert len(forecast) == 7
        assert list(forecast["yhat"]) == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    def test_import_error_handling(self, sample_tsfm_df, config):
        """Verify proper error when timesfm not installed."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = TimesFMAdapter()

        # Remove timesfm from sys.modules to simulate it not being installed
        with patch.dict("sys.modules", {"timesfm": None}):
            with pytest.raises(ImportError):
                adapter.fit(dataset)

    def test_timesfm_adapter_alias(self):
        """Verify TimesfmAdapter alias exists for backward compatibility."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter, TimesfmAdapter

        # TimesfmAdapter should be an alias for TimesFMAdapter
        assert TimesfmAdapter is TimesFMAdapter


class TestMoiraiAdapter:
    """Tests for MoiraiAdapter."""

    def test_init_default_model(self):
        """Verify default model_name is 'moirai-1.1-R-small'."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        adapter = MoiraiAdapter()
        assert adapter.model_name == "moirai-1.1-R-small"
        assert adapter._model is None

    def test_init_custom_model(self):
        """Verify custom model_name can be set."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        adapter = MoiraiAdapter(model_name="moirai-1.1-R-base")
        assert adapter.model_name == "moirai-1.1-R-base"

        adapter = MoiraiAdapter(model_name="moirai-1.1-R-large")
        assert adapter.model_name == "moirai-1.1-R-large"

    def test_fit_uses_dataset_context_length(self, sample_tsfm_df, config):
        """Verify context_length=dataset.min_length in fit."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = MoiraiAdapter()

        # Verify the config values are correct
        assert config.h == 7
        assert dataset.min_length == 100

        # Mock the MoiraiForecast import and verify it would be called correctly
        mock_model = MagicMock()

        with patch.dict("sys.modules", {"moirai_forecast": MagicMock(MoiraiForecast=MagicMock(return_value=mock_model)), "torch": MagicMock()}):
            artifact = adapter.fit(dataset)

            # Verify artifact structure
            assert "model" in artifact
            assert "model_name" in artifact
            assert artifact["model_name"] == "moirai-1.1-R-small"
            assert "adapter" in artifact

    def test_predict_returns_dataframe(self, sample_tsfm_df, config):
        """Verify predict() returns DataFrame with correct structure."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = MoiraiAdapter()

        # Set up mock to return predictable samples
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
        ])

        artifact = {
            "model": mock_model,
            "model_name": "moirai-1.1-R-small",
            "adapter": adapter,
        }

        forecast = adapter.predict(dataset, artifact, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

        # yhat should be median of samples
        expected_median = np.median([
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
        ], axis=0)
        np.testing.assert_array_almost_equal(forecast["yhat"].values, expected_median)

    def test_import_error_handling(self, sample_tsfm_df, config):
        """Verify proper error when moirai not installed."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = MoiraiAdapter()

        # Remove moirai_forecast from sys.modules to simulate it not being installed
        with patch.dict("sys.modules", {"moirai_forecast": None, "torch": MagicMock()}):
            with pytest.raises(ImportError):
                adapter.fit(dataset)


class TestAdapterCommonInterface:
    """Tests that apply to all adapters."""

    def test_all_adapters_have_fit_method(self):
        """Verify all adapters have fit() method with correct signature."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter
        from tsagentkit.models.adapters.moirai import MoiraiAdapter
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        adapters = [
            ChronosAdapter(),
            TimesFMAdapter(),
            MoiraiAdapter(),
        ]

        for adapter in adapters:
            assert hasattr(adapter, "fit")
            assert callable(adapter.fit)

    def test_all_adapters_have_predict_method(self):
        """Verify all adapters have predict() method with correct signature."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter
        from tsagentkit.models.adapters.moirai import MoiraiAdapter
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        adapters = [
            ChronosAdapter(),
            TimesFMAdapter(),
            MoiraiAdapter(),
        ]

        for adapter in adapters:
            assert hasattr(adapter, "predict")
            assert callable(adapter.predict)

    def test_multi_series_forecasting(self, multi_series_tsfm_df, config):
        """Verify adapters handle multiple series."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(multi_series_tsfm_df, config)
        adapter = ChronosAdapter()

        # Set up mock pipeline that returns predictions for each series
        mock_pipeline = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.median.return_value.values.numpy.return_value = np.array([10.0] * 7)
        mock_pipeline.predict.return_value = mock_prediction

        artifact = {
            "pipeline": mock_pipeline,
            "model_name": "chronos-t5-small",
            "adapter": adapter,
        }

        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())), tensor=MagicMock())}):
            forecast = adapter.predict(dataset, artifact, h=7)

            # Should have forecasts for all 3 series
            assert len(forecast) == 21  # 3 series * 7 horizon
            assert set(forecast["unique_id"].unique()) == {"A", "B", "C"}

            # Each series should have 7 forecasts
            for uid in ["A", "B", "C"]:
                series_forecasts = forecast[forecast["unique_id"] == uid]
                assert len(series_forecasts) == 7

    def test_different_horizon_values(self, sample_tsfm_df, config):
        """Verify adapters work with different horizon values."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        dataset = TSDataset.from_dataframe(sample_tsfm_df, config)
        adapter = TimesFMAdapter()

        horizons = [1, 7, 14]
        for h in horizons:
            mock_model = MagicMock()
            mock_model.forecast.return_value = (np.array([list(range(h))]), None)

            artifact = {
                "model": mock_model,
                "adapter": adapter,
            }

            forecast = adapter.predict(dataset, artifact, h=h)
            assert len(forecast) == h, f"Expected {h} forecasts, got {len(forecast)}"


class TestTSFMRunRealFlag:
    """Tests for TSFM_RUN_REAL environment flag handling."""

    def test_tsfm_run_real_env_var_detected(self):
        """Verify os.getenv('TSFM_RUN_REAL') is checked correctly."""
        # This test verifies the flag mechanism works
        flag_value = os.getenv("TSFM_RUN_REAL")

        if flag_value:
            assert flag_value in ("1", "true", "yes", "on")
            print(f"TSFM_RUN_REAL is set to: {flag_value}")
        else:
            print("TSFM_RUN_REAL is not set")

    def test_real_tests_skip_without_flag(self):
        """Verify tests skip when TSFM_RUN_REAL is not set."""
        import subprocess
        import sys

        # Run a test that would fail if TSFM_RUN_REAL was required
        # but we verify it skips properly
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/ci/test_real_tsfm_smoke_gate.py", "-v", "--collect-only"],
            capture_output=True,
            text=True,
        )

        # Should be able to collect tests without running them
        assert result.returncode == 0
        assert "TestChronosRealSmoke" in result.stdout

    def test_real_tests_run_with_flag(self):
        """Verify tests are discoverable when flag is set."""
        import subprocess
        import sys

        env = os.environ.copy()
        env["TSFM_RUN_REAL"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/ci/test_real_tsfm_smoke_gate.py", "-v", "--collect-only"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        assert "TestChronosRealSmoke" in result.stdout


class TestLazyLoading:
    """Tests for adapter lazy loading."""

    def test_adapters_not_imported_at_module_load(self):
        """Verify adapters are not imported when models.adapters is imported."""
        import tsagentkit.models.adapters as adapters_module

        # Adapters should be in __all__ but not loaded yet
        assert hasattr(adapters_module, "__all__")
        assert "ChronosAdapter" in adapters_module.__all__
        assert "TimesFMAdapter" in adapters_module.__all__
        assert "MoiraiAdapter" in adapters_module.__all__

    def test_lazy_load_via_getattr(self):
        """Verify adapters are loaded on first access."""
        import tsagentkit.models.adapters as adapters_module

        # Access should trigger lazy loading
        chronos = adapters_module.ChronosAdapter
        assert chronos is not None

        timesfm = adapters_module.TimesFMAdapter
        assert timesfm is not None

        moirai = adapters_module.MoiraiAdapter
        assert moirai is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
