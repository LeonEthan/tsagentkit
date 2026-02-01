"""Tests for TSFM adapter base classes."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass


# Mock torch before importing adapters
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()
sys.modules["torch.backends.mps"] = MagicMock()

from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry, TSFMAdapter


class MockAdapter(TSFMAdapter):
    """Mock adapter for testing."""

    def load_model(self) -> None:
        self._model = MagicMock()

    def fit(
        self,
        dataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ):
        from tsagentkit.contracts import ModelArtifact

        return ModelArtifact(
            model=self._model,
            model_name="mock",
            config=self.config.__dict__,
        )

    def predict(
        self,
        dataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ):
        from datetime import datetime, timezone

        from tsagentkit.contracts import ForecastResult, Provenance

        # Create mock forecast
        df = pd.DataFrame({
            "unique_id": ["A"] * horizon,
            "ds": pd.date_range("2024-01-01", periods=horizon, freq="D"),
            "yhat": [1.0] * horizon,
        })

        provenance = Provenance(
            run_id="test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_signature="test",
            task_signature="test",
            plan_signature="test",
            model_signature="test",
        )

        return ForecastResult(
            df=df,
            provenance=provenance,
            model_name="mock",
            horizon=horizon,
        )

    def get_model_signature(self) -> str:
        return f"mock-{self.config.model_size}"

    @classmethod
    def _check_dependencies(cls) -> None:
        pass


class TestAdapterConfig:
    """Test AdapterConfig dataclass."""

    def test_default_config(self) -> None:
        config = AdapterConfig(model_name="test")
        assert config.model_name == "test"
        assert config.model_size == "base"
        assert config.device is None
        assert config.batch_size == 32
        assert config.quantile_method == "sample"
        assert config.num_samples == 100

    def test_valid_model_sizes(self) -> None:
        for size in ["tiny", "small", "base", "large"]:
            config = AdapterConfig(model_name="test", model_size=size)
            assert config.model_size == size

    def test_invalid_model_size(self) -> None:
        with pytest.raises(ValueError, match="Invalid model_size"):
            AdapterConfig(model_name="test", model_size="invalid")

    def test_invalid_quantile_method(self) -> None:
        with pytest.raises(ValueError, match="Invalid quantile_method"):
            AdapterConfig(model_name="test", quantile_method="invalid")

    def test_custom_values(self) -> None:
        config = AdapterConfig(
            model_name="chronos",
            model_size="large",
            device="cuda",
            batch_size=64,
            prediction_batch_size=200,
            num_samples=200,
            max_context_length=512,
        )
        assert config.model_name == "chronos"
        assert config.model_size == "large"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.prediction_batch_size == 200
        assert config.num_samples == 200
        assert config.max_context_length == 512


class TestTSFMAdapter:
    """Test TSFMAdapter base class."""

    def test_initialization(self) -> None:
        config = AdapterConfig(model_name="test")
        adapter = MockAdapter(config)

        assert adapter.config == config
        assert adapter._model is None
        assert not adapter.is_loaded

    def test_resolve_device_auto(self) -> None:
        """Test device auto-detection."""
        config = AdapterConfig(model_name="test", device=None)
        adapter = MockAdapter(config)

        # Device should be resolved to one of the valid options
        assert adapter._device in ["cuda", "mps", "cpu"]

    def test_resolve_device_explicit(self) -> None:
        """Test explicit device setting."""
        config = AdapterConfig(model_name="test", device="cuda")
        adapter = MockAdapter(config)
        assert adapter._device == "cuda"

    def test_load_model(self) -> None:
        config = AdapterConfig(model_name="test")
        adapter = MockAdapter(config)

        assert not adapter.is_loaded
        adapter.load_model()
        assert adapter.is_loaded
        assert adapter._model is not None

    def test_unload_model(self) -> None:
        config = AdapterConfig(model_name="test")
        adapter = MockAdapter(config)

        adapter.load_model()
        assert adapter.is_loaded

        adapter.unload_model()
        assert not adapter.is_loaded
        assert adapter._model is None

    def test_predict_creates_valid_result(self) -> None:
        """Test that predict returns a valid ForecastResult."""
        from tsagentkit.contracts import TaskSpec
        from tsagentkit.series import TSDataset

        # Create a simple dataset
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": [1.0] * 10,
        })
        task_spec = TaskSpec(horizon=7, freq="D")
        dataset = TSDataset.from_dataframe(df, task_spec, validate=False)

        config = AdapterConfig(model_name="test")
        adapter = MockAdapter(config)
        adapter.load_model()

        result = adapter.predict(dataset, horizon=7)

        assert result.model_name == "mock"
        assert result.horizon == 7
        assert len(result.df) == 7
        assert "unique_id" in result.df.columns
        assert "ds" in result.df.columns
        assert "yhat" in result.df.columns

    def test_model_signature(self) -> None:
        config = AdapterConfig(model_name="test", model_size="large")
        adapter = MockAdapter(config)
        assert adapter.get_model_signature() == "mock-large"


class TestAdapterRegistry:
    """Test AdapterRegistry functionality."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        AdapterRegistry.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        AdapterRegistry.clear()

    def test_register_and_get(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)
        retrieved = AdapterRegistry.get("mock")
        assert retrieved is MockAdapter

    def test_list_available(self) -> None:
        AdapterRegistry.register("mock1", MockAdapter)
        AdapterRegistry.register("mock2", MockAdapter)

        available = AdapterRegistry.list_available()
        assert available == ["mock1", "mock2"]

    def test_get_unknown_adapter(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter"):
            AdapterRegistry.get("unknown")

    def test_register_duplicate(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)
        # Should not raise when registering same class
        AdapterRegistry.register("mock", MockAdapter)

    def test_register_different_class(self) -> None:
        class AnotherMockAdapter(MockAdapter):
            pass

        AdapterRegistry.register("mock", MockAdapter)
        with pytest.raises(ValueError, match="already registered"):
            AdapterRegistry.register("mock", AnotherMockAdapter)

    def test_unregister(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)
        assert "mock" in AdapterRegistry.list_available()

        AdapterRegistry.unregister("mock")
        assert "mock" not in AdapterRegistry.list_available()

    def test_unregister_unknown(self) -> None:
        with pytest.raises(KeyError):
            AdapterRegistry.unregister("unknown")

    def test_create_adapter(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)
        config = AdapterConfig(model_name="mock", model_size="small")

        adapter = AdapterRegistry.create("mock", config)
        assert isinstance(adapter, MockAdapter)
        assert adapter.config.model_size == "small"

    def test_create_with_default_config(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)

        adapter = AdapterRegistry.create("mock")
        assert isinstance(adapter, MockAdapter)
        assert adapter.config.model_name == "mock"

    def test_check_availability_success(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)
        is_avail, error = AdapterRegistry.check_availability("mock")
        assert is_avail is True
        assert error is None

    def test_check_availability_not_registered(self) -> None:
        is_avail, error = AdapterRegistry.check_availability("unknown")
        assert is_avail is False
        assert "Unknown adapter" in error

    def test_check_availability_missing_deps(self) -> None:
        class BadAdapter(MockAdapter):
            @classmethod
            def _check_dependencies(cls) -> None:
                raise ImportError("Missing dependency")

        AdapterRegistry.register("bad", BadAdapter)
        is_avail, error = AdapterRegistry.check_availability("bad")
        assert is_avail is False
        assert "Missing dependency" in error

    def test_get_available_adapters(self) -> None:
        class BadAdapter(MockAdapter):
            @classmethod
            def _check_dependencies(cls) -> None:
                raise ImportError("Missing dependency")

        AdapterRegistry.register("mock", MockAdapter)
        AdapterRegistry.register("bad", BadAdapter)

        available = AdapterRegistry.get_available_adapters()
        assert "mock" in available
        assert "bad" not in available

    def test_clear_registry(self) -> None:
        AdapterRegistry.register("mock", MockAdapter)
        assert len(AdapterRegistry.list_available()) == 1

        AdapterRegistry.clear()
        assert len(AdapterRegistry.list_available()) == 0
