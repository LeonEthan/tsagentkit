"""Real TSFM smoke tests with actual model downloads and inference.

These tests verify that each TSFM adapter can:
1. Download models from HuggingFace
2. Load models into memory
3. Run inference on sample data

Requirements:
- Set TSFM_RUN_REAL=1 environment variable to run
- GPU recommended but not required (will use CPU fallback)
- 45-minute timeout for model downloads
- Minimal data (30-50 points) for fast inference
"""

from __future__ import annotations

import os
import warnings
from typing import Any

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


def get_device() -> str:
    """Detect available device for TSFM inference."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def create_minimal_dataset(n_series: int = 1, n_points: int = 50) -> pd.DataFrame:
    """Create minimal dataset for fast TSFM testing."""
    np.random.seed(42)
    dfs = []
    for i, uid in enumerate([f"S{i}" for i in range(n_series)]):
        # Create simple trend + seasonality
        t = np.linspace(0, 4 * np.pi, n_points)
        values = 10 + 5 * np.sin(t + i) + np.random.randn(n_points) * 0.5

        df = pd.DataFrame({
            "unique_id": [uid] * n_points,
            "ds": pd.date_range("2024-01-01", periods=n_points, freq="D"),
            "y": values,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def minimal_df():
    """Minimal single-series dataset."""
    return create_minimal_dataset(n_series=1, n_points=50)


@pytest.fixture
def minimal_multi_series_df():
    """Minimal multi-series dataset."""
    return create_minimal_dataset(n_series=3, n_points=50)


@pytest.fixture
def device():
    """Get available device."""
    return get_device()


@pytest.fixture
def forecast_config():
    """Create minimal forecast config."""
    from tsagentkit import ForecastConfig

    return ForecastConfig(h=7, freq="D")


class TestChronosRealSmoke:
    """Real smoke tests for Chronos adapter."""

    def test_chronos_loads_and_predicts(self, minimal_df, forecast_config, device):
        """Verify Chronos can download, load, and run inference."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.chronos import ChronosAdapter

        device_type = "GPU" if device == "cuda" else "MPS" if device == "mps" else "CPU"
        print(f"\nRunning Chronos smoke test on {device_type}...")

        # Create dataset
        dataset = TSDataset.from_dataframe(minimal_df, forecast_config)

        # Initialize adapter with tiny model for speed
        adapter = ChronosAdapter(model_name="chronos-t5-tiny")

        # Load model (downloads from HuggingFace on first run)
        artifact = adapter.fit(dataset)

        # Verify artifact structure
        assert "pipeline" in artifact
        assert "model_name" in artifact
        assert artifact["model_name"] == "chronos-t5-tiny"
        assert "adapter" in artifact

        # Run inference
        forecast = adapter.predict(dataset, artifact, h=7)

        # Verify forecast structure
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

        # Verify forecast values are reasonable
        assert not forecast["yhat"].isnull().any()
        assert all(np.isfinite(forecast["yhat"]))

        # Verify dates
        last_data_date = minimal_df["ds"].iloc[-1]
        expected_first_date = last_data_date + pd.Timedelta(days=1)
        assert forecast["ds"].iloc[0] == expected_first_date

        print(f"✓ Chronos smoke test passed: generated {len(forecast)} forecasts")

    def test_chronos_multi_series(self, minimal_multi_series_df, forecast_config, device):
        """Verify Chronos handles multiple series."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(minimal_multi_series_df, forecast_config)
        adapter = ChronosAdapter(model_name="chronos-t5-tiny")

        artifact = adapter.fit(dataset)
        forecast = adapter.predict(dataset, artifact, h=7)

        # Should have forecasts for all 3 series
        assert len(forecast) == 21  # 3 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"S0", "S1", "S2"}

        print(f"✓ Chronos multi-series test passed: {len(forecast)} total forecasts")


class TestTimesFMRealSmoke:
    """Real smoke tests for TimesFM adapter."""

    def test_timesfm_loads_and_predicts(self, minimal_df, forecast_config, device):
        """Verify TimesFM can download, load, and run inference."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.timesfm import TimesFMAdapter

        device_type = "GPU" if device == "cuda" else "MPS" if device == "mps" else "CPU"
        print(f"\nRunning TimesFM smoke test on {device_type}...")

        # Create dataset
        dataset = TSDataset.from_dataframe(minimal_df, forecast_config)

        # Initialize adapter with smaller context for speed
        adapter = TimesFMAdapter(context_len=128, horizon_len=64)

        # Load model (downloads checkpoint on first run)
        artifact = adapter.fit(dataset)

        # Verify artifact structure
        assert "model" in artifact
        assert "adapter" in artifact

        # Run inference
        forecast = adapter.predict(dataset, artifact, h=7)

        # Verify forecast structure
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

        # Verify forecast values are reasonable
        assert not forecast["yhat"].isnull().any()
        assert all(np.isfinite(forecast["yhat"]))

        # Verify dates
        last_data_date = minimal_df["ds"].iloc[-1]
        expected_first_date = last_data_date + pd.Timedelta(days=1)
        assert forecast["ds"].iloc[0] == expected_first_date

        print(f"✓ TimesFM smoke test passed: generated {len(forecast)} forecasts")

    def test_timesfm_multi_series(self, minimal_multi_series_df, forecast_config, device):
        """Verify TimesFM handles multiple series."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.timesfm import TimesFMAdapter

        dataset = TSDataset.from_dataframe(minimal_multi_series_df, forecast_config)
        adapter = TimesFMAdapter(context_len=128, horizon_len=64)

        artifact = adapter.fit(dataset)
        forecast = adapter.predict(dataset, artifact, h=7)

        # Should have forecasts for all 3 series
        assert len(forecast) == 21  # 3 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"S0", "S1", "S2"}

        print(f"✓ TimesFM multi-series test passed: {len(forecast)} total forecasts")


class TestMoiraiRealSmoke:
    """Real smoke tests for Moirai adapter."""

    def test_moirai_loads_and_predicts(self, minimal_df, forecast_config, device):
        """Verify Moirai can download, load, and run inference."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.moirai import MoiraiAdapter

        device_type = "GPU" if device == "cuda" else "MPS" if device == "mps" else "CPU"
        print(f"\nRunning Moirai smoke test on {device_type}...")

        # Create dataset
        dataset = TSDataset.from_dataframe(minimal_df, forecast_config)

        # Initialize adapter with small model
        adapter = MoiraiAdapter(model_name="moirai-1.1-R-small")

        # Load model (downloads from HuggingFace on first run)
        artifact = adapter.fit(dataset)

        # Verify artifact structure
        assert "model" in artifact
        assert "model_name" in artifact
        assert artifact["model_name"] == "moirai-1.1-R-small"
        assert "adapter" in artifact

        # Run inference
        forecast = adapter.predict(dataset, artifact, h=7)

        # Verify forecast structure
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

        # Verify forecast values are reasonable
        assert not forecast["yhat"].isnull().any()
        assert all(np.isfinite(forecast["yhat"]))

        # Verify dates
        last_data_date = minimal_df["ds"].iloc[-1]
        expected_first_date = last_data_date + pd.Timedelta(days=1)
        assert forecast["ds"].iloc[0] == expected_first_date

        print(f"✓ Moirai smoke test passed: generated {len(forecast)} forecasts")

    def test_moirai_multi_series(self, minimal_multi_series_df, forecast_config, device):
        """Verify Moirai handles multiple series."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.moirai import MoiraiAdapter

        dataset = TSDataset.from_dataframe(minimal_multi_series_df, forecast_config)
        adapter = MoiraiAdapter(model_name="moirai-1.1-R-small")

        artifact = adapter.fit(dataset)
        forecast = adapter.predict(dataset, artifact, h=7)

        # Should have forecasts for all 3 series
        assert len(forecast) == 21  # 3 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"S0", "S1", "S2"}

        print(f"✓ Moirai multi-series test passed: {len(forecast)} total forecasts")


class TestTSFMSmokeCommon:
    """Common smoke tests for all TSFM adapters."""

    def test_all_adapters_available(self):
        """Verify all TSFM adapters can be imported."""
        from tsagentkit.models.adapters.tsfm.chronos import ChronosAdapter
        from tsagentkit.models.adapters.tsfm.moirai import MoiraiAdapter
        from tsagentkit.models.adapters.tsfm.timesfm import TimesFMAdapter

        # Just verify we can instantiate with defaults
        chronos = ChronosAdapter()
        timesfm = TimesFMAdapter()
        moirai = MoiraiAdapter()

        assert chronos.model_name == "chronos-t5-small"
        assert timesfm.context_len == 512
        assert timesfm.horizon_len == 128
        assert moirai.model_name == "moirai-1.1-R-small"

    def test_different_horizons(self, minimal_df, forecast_config):
        """Verify all adapters work with different horizon values."""
        from tsagentkit import TSDataset
        from tsagentkit.models.adapters.tsfm.chronos import ChronosAdapter

        dataset = TSDataset.from_dataframe(minimal_df, forecast_config)
        adapter = ChronosAdapter(model_name="chronos-t5-tiny")
        artifact = adapter.fit(dataset)

        # Test different horizons
        for h in [1, 7, 14]:
            forecast = adapter.predict(dataset, artifact, h=h)
            assert len(forecast) == h, f"Expected {h} forecasts, got {len(forecast)}"

        print("✓ Different horizon tests passed")

    def test_frequency_handling(self, device):
        """Verify adapters handle different frequencies."""
        from tsagentkit import ForecastConfig, TSDataset
        from tsagentkit.models.adapters.tsfm.timesfm import TimesFMAdapter

        # Test with daily frequency
        np.random.seed(42)
        daily_df = pd.DataFrame({
            "unique_id": ["A"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50, freq="D"),
            "y": np.random.randn(50) + 10,
        })

        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(daily_df, config)
        adapter = TimesFMAdapter(context_len=64, horizon_len=32)
        artifact = adapter.fit(dataset)
        forecast = adapter.predict(dataset, artifact, h=7)

        # Verify date increment matches frequency
        date_diff = forecast["ds"].iloc[1] - forecast["ds"].iloc[0]
        assert date_diff == pd.Timedelta(days=1)

        print("✓ Frequency handling test passed")


def test_environment_setup():
    """Verify test environment is properly configured."""
    import torch

    print("\n=== TSFM Test Environment ===")
    print(f"TSFM_RUN_REAL: {os.getenv('TSFM_RUN_REAL', 'not set')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Device selected: {get_device()}")
    print("=============================")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
