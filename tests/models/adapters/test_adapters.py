"""Tests for TSFM adapters (Chronos, Moirai, TimesFM).

These tests use mocking to avoid dependencies on the actual TSFM packages.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    pass


# Setup mocks before importing adapters
@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup mocks for TSFM dependencies."""
    # Mock torch
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.backends = MagicMock()
    torch_mock.backends.mps = MagicMock()
    torch_mock.backends.mps.is_available = MagicMock(return_value=False)
    torch_mock.device = MagicMock(return_value="cpu")

    # Mock chronos (Chronos2)
    chronos_mock = MagicMock()
    chronos2_pipeline_mock = MagicMock()
    chronos2_pipeline_mock.from_pretrained = MagicMock(return_value=MagicMock())
    chronos_mock.Chronos2Pipeline = chronos2_pipeline_mock

    # Mock uni2ts
    uni2ts_mock = MagicMock()
    moirai_module_mock = MagicMock()
    moirai_module_mock.from_pretrained = MagicMock(return_value=MagicMock())
    moirai_forecast_mock = MagicMock()
    uni2ts_mock.model = MagicMock()
    uni2ts_mock.model.moirai = MagicMock()
    uni2ts_mock.model.moirai.MoiraiModule = moirai_module_mock
    uni2ts_mock.model.moirai.MoiraiForecast = moirai_forecast_mock

    # Mock timesfm
    timesfm_mock = MagicMock()
    timesfm_model_cls = MagicMock()
    timesfm_model_cls.from_pretrained = MagicMock(return_value=MagicMock())
    timesfm_mock.TimesFM_2p5_200M_torch = timesfm_model_cls
    timesfm_mock.ForecastConfig = MagicMock()

    with patch.dict(sys.modules, {
        "torch": torch_mock,
        "chronos": chronos_mock,
        "uni2ts": uni2ts_mock,
        "uni2ts.model": uni2ts_mock.model,
        "uni2ts.model.moirai": uni2ts_mock.model.moirai,
        "timesfm": timesfm_mock,
    }):
        yield


# Import after mocks are set up
from tsagentkit.contracts import TaskSpec
from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry
from tsagentkit.series import TSDataset


class TestChronosAdapter:
    """Test Chronos adapter."""

    def test_adapter_exists(self) -> None:
        """Test that ChronosAdapter can be imported."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter
        assert ChronosAdapter is not None

    def test_model_sizes(self) -> None:
        """Test model size configuration."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        expected_sizes = ["small", "base"]
        for size in expected_sizes:
            assert size in ChronosAdapter.MODEL_SIZES
            assert "chronos" in ChronosAdapter.MODEL_SIZES[size]

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        config = AdapterConfig(model_name="chronos", model_size="base")
        adapter = ChronosAdapter(config)

        assert adapter.config == config
        assert not adapter.is_loaded

    def test_get_model_signature(self) -> None:
        """Test model signature generation."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        config = AdapterConfig(model_name="chronos", model_size="large", device="cuda")
        adapter = ChronosAdapter(config)

        signature = adapter.get_model_signature()
        assert "chronos" in signature
        assert "large" in signature
        assert "cuda" in signature


class TestMoiraiAdapter:
    """Test Moirai adapter."""

    def test_adapter_exists(self) -> None:
        """Test that MoiraiAdapter can be imported."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter
        assert MoiraiAdapter is not None

    def test_model_id(self) -> None:
        """Test model ID constant."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        assert MoiraiAdapter.MODEL_ID == "Salesforce/moirai-2.0-R-small"
        assert "moirai-2.0" in MoiraiAdapter.MODEL_ID

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        # model_size is ignored for Moirai 2.0 (only one model exists)
        config = AdapterConfig(model_name="moirai", model_size="small")
        adapter = MoiraiAdapter(config)

        assert adapter.config == config
        assert not adapter.is_loaded

    def test_context_length(self) -> None:
        """Test context length defaults."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        assert MoiraiAdapter.DEFAULT_CONTEXT_LENGTH == 512


class TestTimesFMAdapter:
    """Test TimesFM adapter."""

    def test_adapter_exists(self) -> None:
        """Test that TimesFMAdapter can be imported."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter
        assert TimesFMAdapter is not None

    def test_model_id(self) -> None:
        """Test model ID constant."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        assert TimesFMAdapter.MODEL_ID == "google/timesfm-2.5-200m-pytorch"
        assert "timesfm-2.5" in TimesFMAdapter.MODEL_ID

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        # model_size is ignored for TimesFM 2.5 (only one model exists)
        config = AdapterConfig(model_name="timesfm", model_size="base")
        adapter = TimesFMAdapter(config)

        assert adapter.config == config
        assert not adapter.is_loaded

    def test_get_model_signature(self) -> None:
        """Test model signature generation."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        config = AdapterConfig(model_name="timesfm", device="cpu")
        adapter = TimesFMAdapter(config)

        signature = adapter.get_model_signature()
        assert "timesfm-2.5" in signature
        assert "cpu" in signature


class TestAdapterRegistration:
    """Test that adapters can be registered."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        AdapterRegistry.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        AdapterRegistry.clear()

    def test_register_chronos(self) -> None:
        """Test registering Chronos adapter."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        AdapterRegistry.register("chronos", ChronosAdapter)
        assert "chronos" in AdapterRegistry.list_available()

        # With mocks, chronos should be available
        is_avail, error = AdapterRegistry.check_availability("chronos")
        assert is_avail
        assert error == ""

    def test_register_moirai(self) -> None:
        """Test registering Moirai adapter."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        AdapterRegistry.register("moirai", MoiraiAdapter)
        assert "moirai" in AdapterRegistry.list_available()

    def test_register_timesfm(self) -> None:
        """Test registering TimesFM adapter."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        AdapterRegistry.register("timesfm", TimesFMAdapter)
        assert "timesfm" in AdapterRegistry.list_available()


class TestAdapterHandleMissingValues:
    """Test missing value handling in adapters."""

    def test_chronos_handle_missing(self) -> None:
        """Test Chronos missing value handling."""
        from tsagentkit.models.adapters.chronos import ChronosAdapter

        config = AdapterConfig(model_name="chronos")
        adapter = ChronosAdapter(config)

        # Create array with NaN
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        filled = adapter._handle_missing_values(values)

        assert not np.any(np.isnan(filled))
        assert filled[0] == 1.0
        assert filled[2] == 3.0
        assert filled[4] == 5.0

    def test_moirai_handle_missing(self) -> None:
        """Test Moirai missing value handling."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        config = AdapterConfig(model_name="moirai")
        adapter = MoiraiAdapter(config)

        values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        filled = adapter._handle_missing_values(values)

        assert not np.any(np.isnan(filled))

    def test_timesfm_handle_missing(self) -> None:
        """Test TimesFM missing value handling."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        config = AdapterConfig(model_name="timesfm")
        adapter = TimesFMAdapter(config)

        values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        filled = adapter._handle_missing_values(values)

        assert not np.any(np.isnan(filled))
