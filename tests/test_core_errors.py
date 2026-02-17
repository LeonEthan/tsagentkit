"""Tests for core error types.

Tests the error hierarchy and rich context functionality.
"""

from __future__ import annotations

import pytest

from tsagentkit.core.errors import (
    EContract,
    EInsufficient,
    ENoTSFM,
    ETemporal,
    TSAgentKitError,
)


class TestTSAgentKitError:
    """Test base error class."""

    def test_basic_error(self):
        """Basic error creation."""
        err = TSAgentKitError("Something went wrong")
        assert err.message == "Something went wrong"
        assert err.error_code == "E_UNKNOWN"
        assert err.context == {}

    def test_error_with_context(self):
        """Error with context."""
        context = {"series_id": "A", "value": 42}
        err = TSAgentKitError("Test error", context=context)
        assert err.context == context
        assert "series_id" in str(err)

    def test_error_with_fix_hint(self):
        """Error with fix hint."""
        err = TSAgentKitError("Test error", fix_hint="Try restarting")
        assert err.fix_hint == "Try restarting"
        assert "Try restarting" in str(err)

    def test_error_str_format(self):
        """Error string formatting."""
        err = TSAgentKitError("Test message", context={"key": "value"}, fix_hint="Do this")
        err_str = str(err)
        assert "[E_UNKNOWN]" in err_str
        assert "Test message" in err_str
        assert "key" in err_str
        assert "Do this" in err_str


class TestEContract:
    """Test contract error."""

    def test_error_code(self):
        """Error has correct code."""
        err = EContract("Missing column")
        assert err.error_code == "E_CONTRACT"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = EContract("Missing column")
        assert "unique_id" in err.fix_hint
        assert "ds" in err.fix_hint
        assert "y" in err.fix_hint

    def test_custom_fix_hint(self):
        """Can override fix hint."""
        err = EContract("Missing column", fix_hint="Custom hint")
        assert err.fix_hint == "Custom hint"


class TestENoTSFM:
    """Test no TSFM error."""

    def test_error_code(self):
        """Error has correct code."""
        err = ENoTSFM("No TSFM available")
        assert err.error_code == "E_NO_TSFM"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = ENoTSFM("No TSFM available")
        assert "chronos" in err.fix_hint or "TSFM" in err.fix_hint


class TestEInsufficient:
    """Test insufficient TSFMs error."""

    def test_error_code(self):
        """Error has correct code."""
        err = EInsufficient("Not enough models")
        assert err.error_code == "E_INSUFFICIENT"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = EInsufficient("Not enough models")
        assert "compatibility" in err.fix_hint or "min_tsfm" in err.fix_hint


class TestETemporal:
    """Test temporal integrity error."""

    def test_error_code(self):
        """Error has correct code."""
        err = ETemporal("Data not sorted")
        assert err.error_code == "E_TEMPORAL"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = ETemporal("Data not sorted")
        assert "sorted" in err.fix_hint or "ds" in err.fix_hint


class TestErrorRaising:
    """Test that errors can be raised and caught."""

    def test_raise_catch_contract(self):
        """Can raise and catch contract error."""
        with pytest.raises(EContract) as exc_info:
            raise EContract("Test")
        assert exc_info.value.error_code == "E_CONTRACT"

    def test_raise_catch_no_tsfm(self):
        """Can raise and catch no TSFM error."""
        with pytest.raises(ENoTSFM) as exc_info:
            raise ENoTSFM("Test")
        assert exc_info.value.error_code == "E_NO_TSFM"

    def test_raise_catch_insufficient(self):
        """Can raise and catch insufficient error."""
        with pytest.raises(EInsufficient) as exc_info:
            raise EInsufficient("Test")
        assert exc_info.value.error_code == "E_INSUFFICIENT"

    def test_raise_catch_temporal(self):
        """Can raise and catch temporal error."""
        with pytest.raises(ETemporal) as exc_info:
            raise ETemporal("Test")
        assert exc_info.value.error_code == "E_TEMPORAL"

    def test_catch_base_exception(self):
        """All errors catchable as base exception."""
        with pytest.raises(TSAgentKitError):
            raise EContract("Test")
        with pytest.raises(TSAgentKitError):
            raise ENoTSFM("Test")
        with pytest.raises(TSAgentKitError):
            raise EInsufficient("Test")
        with pytest.raises(TSAgentKitError):
            raise ETemporal("Test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
