"""Tests for contracts/errors.py."""

import pytest

from tsagentkit.contracts import (
    EContractDuplicateKey,
    EContractInvalidFrequency,
    EContractInvalidType,
    EContractMissingColumn,
    EContractUnsorted,
    ECovariateLeakage,
    EFallbackExhausted,
    EModelFitFailed,
    ESplitRandomForbidden,
    TSAgentKitError,
    get_error_class,
)


class TestTSAgentKitError:
    """Tests for base error class."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        err = TSAgentKitError("Something went wrong")
        assert err.message == "Something went wrong"
        assert err.error_code == "E_UNKNOWN"
        assert str(err) == "[E_UNKNOWN] Something went wrong"

    def test_error_with_context(self) -> None:
        """Test error with context."""
        err = TSAgentKitError("Validation failed", {"field": "value"})
        assert err.context == {"field": "value"}
        assert "field" in str(err)


class TestSpecificErrors:
    """Tests for specific error classes."""

    def test_error_codes(self) -> None:
        """Test that each error has correct error code."""
        errors = [
            (EContractMissingColumn, "E_CONTRACT_MISSING_COLUMN"),
            (EContractInvalidType, "E_CONTRACT_INVALID_TYPE"),
            (EContractDuplicateKey, "E_CONTRACT_DUPLICATE_KEY"),
            (EContractUnsorted, "E_CONTRACT_UNSORTED"),
            (EContractInvalidFrequency, "E_CONTRACT_INVALID_FREQUENCY"),
            (ESplitRandomForbidden, "E_SPLIT_RANDOM_FORBIDDEN"),
            (ECovariateLeakage, "E_COVARIATE_LEAKAGE"),
            (EModelFitFailed, "E_MODEL_FIT_FAILED"),
            (EFallbackExhausted, "E_FALLBACK_EXHAUSTED"),
        ]

        for err_class, expected_code in errors:
            assert err_class.error_code == expected_code
            err = err_class("Test message")
            assert err.error_code == expected_code
            assert expected_code in str(err)

    def test_inheritance(self) -> None:
        """Test that all errors inherit from TSAgentKitError."""
        errors = [
            EContractMissingColumn,
            ESplitRandomForbidden,
            ECovariateLeakage,
        ]

        for err_class in errors:
            err = err_class("Test")
            assert isinstance(err, TSAgentKitError)


class TestErrorRegistry:
    """Tests for error registry."""

    def test_get_error_class(self) -> None:
        """Test lookup by error code."""
        assert get_error_class("E_CONTRACT_MISSING_COLUMN") == EContractMissingColumn
        assert get_error_class("E_SPLIT_RANDOM_FORBIDDEN") == ESplitRandomForbidden

    def test_get_unknown_error(self) -> None:
        """Test lookup returns base class for unknown code."""
        assert get_error_class("E_UNKNOWN_CODE") == TSAgentKitError
