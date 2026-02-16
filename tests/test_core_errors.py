"""Tests for core error types.

Tests the error hierarchy and rich context functionality.
"""

from __future__ import annotations

import pytest

from tsagentkit.core.errors import (
    ERROR_REGISTRY,
    EContractViolation,
    EDataQuality,
    EModelFailed,
    ETSFMRequired,
    TSAgentKitError,
    get_error_class,
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


class TestEContractViolation:
    """Test contract violation error."""

    def test_error_code(self):
        """Error has correct code."""
        err = EContractViolation("Missing column")
        assert err.error_code == "E_CONTRACT_VIOLATION"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = EContractViolation("Missing column")
        assert "unique_id" in err.fix_hint
        assert "ds" in err.fix_hint
        assert "y" in err.fix_hint

    def test_custom_fix_hint(self):
        """Can override fix hint."""
        err = EContractViolation("Missing column", fix_hint="Custom hint")
        assert err.fix_hint == "Custom hint"


class TestEDataQuality:
    """Test data quality error."""

    def test_error_code(self):
        """Error has correct code."""
        err = EDataQuality("Duplicates found")
        assert err.error_code == "E_DATA_QUALITY"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = EDataQuality("Duplicates found")
        assert "diagnose()" in err.fix_hint


class TestEModelFailed:
    """Test model failed error."""

    def test_error_code(self):
        """Error has correct code."""
        err = EModelFailed("Model crashed")
        assert err.error_code == "E_MODEL_FAILED"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = EModelFailed("Model crashed")
        assert "compatibility" in err.fix_hint


class TestETSFMRequired:
    """Test TSFM required error."""

    def test_error_code(self):
        """Error has correct code."""
        err = ETSFMRequired("No TSFM available")
        assert err.error_code == "E_TSFM_REQUIRED"

    def test_default_fix_hint(self):
        """Has default fix hint."""
        err = ETSFMRequired("No TSFM available")
        assert "chronos" in err.fix_hint
        assert "moirai" in err.fix_hint
        assert "timesfm" in err.fix_hint


class TestErrorRegistry:
    """Test error registry functionality."""

    def test_registry_contains_all_errors(self):
        """Registry contains all error types."""
        assert "E_CONTRACT_VIOLATION" in ERROR_REGISTRY
        assert "E_DATA_QUALITY" in ERROR_REGISTRY
        assert "E_MODEL_FAILED" in ERROR_REGISTRY
        assert "E_TSFM_REQUIRED" in ERROR_REGISTRY

    def test_get_error_class(self):
        """Can retrieve error class by code."""
        cls = get_error_class("E_CONTRACT_VIOLATION")
        assert cls == EContractViolation

    def test_get_error_class_unknown(self):
        """Unknown code returns base TSAgentKitError."""
        cls = get_error_class("E_UNKNOWN_CODE")
        assert cls == TSAgentKitError

    def test_registry_values_are_error_classes(self):
        """All registry values are error classes."""
        for code, cls in ERROR_REGISTRY.items():
            assert issubclass(cls, TSAgentKitError)
            assert cls.error_code == code


class TestLegacyAliases:
    """Test backward compatibility aliases."""

    def test_legacy_contract_aliases(self):
        """Legacy contract error aliases work."""
        from tsagentkit.core.errors import (
            EContractDuplicateKey,
            EContractInvalidType,
            EContractMissingColumn,
            EFreqInferFail,
            ESplitRandomForbidden,
        )
        assert EContractMissingColumn == EContractViolation
        assert EContractInvalidType == EContractViolation
        assert EContractDuplicateKey == EContractViolation
        assert EFreqInferFail == EContractViolation
        assert ESplitRandomForbidden == EContractViolation

    def test_legacy_quality_aliases(self):
        """Legacy quality error aliases work."""
        from tsagentkit.core.errors import (
            ECovariateLeakage,
            EQACriticalIssue,
            EQAMinHistory,
        )
        assert EQAMinHistory == EDataQuality
        assert EQACriticalIssue == EDataQuality
        assert ECovariateLeakage == EDataQuality

    def test_legacy_model_aliases(self):
        """Legacy model error aliases work."""
        from tsagentkit.core.errors import (
            EFallbackExhausted,
            EModelFitFailed,
            EModelPredictFailed,
        )
        assert EModelFitFailed == EModelFailed
        assert EModelPredictFailed == EModelFailed
        assert EFallbackExhausted == EModelFailed

    def test_legacy_tsfm_aliases(self):
        """Legacy TSFM error aliases work."""
        from tsagentkit.core.errors import ETSFMRequiredUnavailable
        assert ETSFMRequiredUnavailable == ETSFMRequired


class TestErrorRaising:
    """Test that errors can be raised and caught."""

    def test_raise_catch_contract(self):
        """Can raise and catch contract violation."""
        with pytest.raises(EContractViolation) as exc_info:
            raise EContractViolation("Test")
        assert exc_info.value.error_code == "E_CONTRACT_VIOLATION"

    def test_raise_catch_data_quality(self):
        """Can raise and catch data quality error."""
        with pytest.raises(EDataQuality) as exc_info:
            raise EDataQuality("Test")
        assert exc_info.value.error_code == "E_DATA_QUALITY"

    def test_raise_catch_model_failed(self):
        """Can raise and catch model failed error."""
        with pytest.raises(EModelFailed) as exc_info:
            raise EModelFailed("Test")
        assert exc_info.value.error_code == "E_MODEL_FAILED"

    def test_raise_catch_tsfm_required(self):
        """Can raise and catch TSFM required error."""
        with pytest.raises(ETSFMRequired) as exc_info:
            raise ETSFMRequired("Test")
        assert exc_info.value.error_code == "E_TSFM_REQUIRED"

    def test_catch_base_exception(self):
        """All errors catchable as base exception."""
        with pytest.raises(TSAgentKitError):
            raise EContractViolation("Test")
        with pytest.raises(TSAgentKitError):
            raise EDataQuality("Test")
        with pytest.raises(TSAgentKitError):
            raise EModelFailed("Test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
