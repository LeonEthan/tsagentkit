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

    # Mock chronos
    chronos_mock = MagicMock()
    chronos_pipeline_mock = MagicMock()
    chronos_pipeline_mock.from_pretrained = MagicMock(return_value=MagicMock())
    chronos_mock.ChronosPipeline = chronos_pipeline_mock

    # Mock uni2ts
    uni2ts_mock = MagicMock()
    moirai_mock = MagicMock()
    moirai_mock.load_from_checkpoint = MagicMock(return_value=MagicMock())
    uni2ts_mock.model = MagicMock()
    uni2ts_mock.model.moirai = MagicMock()
    uni2ts_mock.model.moirai.MoiraiForecast = moirai_mock

    # Mock timesfm
    timesfm_mock = MagicMock()
    timesfm_model_mock = MagicMock()
    timesfm_model_mock.load_from_checkpoint = MagicMock()
    timesfm_mock.TimesFm = MagicMock(return_value=timesfm_model_mock)

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

        expected_sizes = ["tiny", "small", "base", "large"]
        for size in expected_sizes:
            assert size in ChronosAdapter.MODEL_SIZES
            assert ChronosAdapter.MODEL_SIZES[size].startswith("amazon/chronos")

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

    def test_model_sizes(self) -> None:
        """Test model size configuration."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        expected_sizes = ["small", "base", "large"]
        for size in expected_sizes:
            assert size in MoiraiAdapter.MODEL_SIZES
            assert MoiraiAdapter.MODEL_SIZES[size].startswith("Salesforce/moirai")

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        config = AdapterConfig(model_name="moirai", model_size="base")
        adapter = MoiraiAdapter(config)

        assert adapter.config == config
        assert not adapter.is_loaded

    def test_get_patch_size(self) -> None:
        """Test patch size selection."""
        from tsagentkit.models.adapters.moirai import MoiraiAdapter

        config = AdapterConfig(model_name="moirai", model_size="base")
        adapter = MoiraiAdapter(config)

        assert adapter._get_patch_size("D") == 32
        assert adapter._get_patch_size("H") == 24
        assert adapter._get_patch_size("W") == 4


class TestTimesFMAdapter:
    """Test TimesFM adapter."""

    def test_adapter_exists(self) -> None:
        """Test that TimesFMAdapter can be imported."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter
        assert TimesFMAdapter is not None

    def test_model_sizes(self) -> None:
        """Test model size configuration."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        expected_sizes = ["base", "large"]
        for size in expected_sizes:
            assert size in TimesFMAdapter.MODEL_SIZES
            assert TimesFMAdapter.MODEL_SIZES[size].startswith("google/timesfm")

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        config = AdapterConfig(model_name="timesfm", model_size="base")
        adapter = TimesFMAdapter(config)

        assert adapter.config == config
        assert not adapter.is_loaded

    def test_map_frequency(self) -> None:
        """Test frequency mapping."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        config = AdapterConfig(model_name="timesfm")
        adapter = TimesFMAdapter(config)

        assert adapter._map_frequency("D") == "D"
        assert adapter._map_frequency("H") == "H"
        assert adapter._map_frequency("W") == "W"
        assert adapter._map_frequency("M") == "M"
        assert adapter._map_frequency("Q") == "Q"
        assert adapter._map_frequency("Y") == "Y"
        # Test unknown defaults to D
        assert adapter._map_frequency("X") == "D"

    def test_get_model_signature(self) -> None:
        """Test model signature generation."""
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter

        config = AdapterConfig(model_name="timesfm", model_size="base", device="cpu")
        adapter = TimesFMAdapter(config)

        signature = adapter.get_model_signature()
        assert "timesfm" in signature
        assert "base" in signature
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
        assert error is None

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
