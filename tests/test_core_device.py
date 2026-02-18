"""Tests for device resolution utility."""

from __future__ import annotations

import pytest

from tsagentkit.core.device import resolve_device


class TestResolveDevice:
    """Test suite for resolve_device function."""

    def test_resolve_cpu_explicit(self):
        """Explicit cpu should always return cpu."""
        result = resolve_device("cpu")
        assert result == "cpu"

    def test_resolve_auto_returns_valid_device(self):
        """Auto should return a valid device string."""
        result = resolve_device("auto")
        assert result in ("cuda", "mps", "cpu")

    def test_resolve_auto_no_mps_returns_valid_device(self):
        """Auto with allow_mps=False should return cuda or cpu."""
        result = resolve_device("auto", allow_mps=False)
        assert result in ("cuda", "cpu")

    def test_resolve_cuda_returns_cpu_when_unavailable(self):
        """Explicit cuda should fallback to cpu when unavailable."""
        # We can't control hardware, but we can verify it returns a valid device
        result = resolve_device("cuda")
        assert result in ("cuda", "cpu")

    def test_resolve_mps_returns_cpu_when_unavailable(self):
        """Explicit mps should fallback to cpu when unavailable."""
        result = resolve_device("mps")
        assert result in ("mps", "cpu")


class TestDeviceConfigIntegration:
    """Test device parameter integration with ForecastConfig."""

    def test_config_accepts_auto_device(self):
        """ForecastConfig should accept device='auto'."""
        from tsagentkit.core.config import ForecastConfig

        config = ForecastConfig(h=7, device="auto")
        assert config.device == "auto"

    def test_config_accepts_explicit_devices(self):
        """ForecastConfig should accept explicit device values."""
        from tsagentkit.core.config import ForecastConfig

        config_cpu = ForecastConfig(h=7, device="cpu")
        config_cuda = ForecastConfig(h=7, device="cuda")
        config_mps = ForecastConfig(h=7, device="mps")

        assert config_cpu.device == "cpu"
        assert config_cuda.device == "cuda"
        assert config_mps.device == "mps"

    def test_config_default_device_is_auto(self):
        """ForecastConfig default device should be 'auto'."""
        from tsagentkit.core.config import ForecastConfig

        config = ForecastConfig(h=7)
        assert config.device == "auto"
