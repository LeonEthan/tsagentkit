"""Tests for WS-3.1/3.2: fix_hint on TSAgentKitError and subclasses."""

from __future__ import annotations

import pytest

from tsagentkit.contracts.errors import (
    EBacktestInsufficientData,
    ECalibrationFail,
    EContractDuplicateKey,
    EContractMissingColumn,
    ECovariateLeakage,
    EDSNotMonotonic,
    EFallbackExhausted,
    EFreqInferFail,
    EQAMinHistory,
    ETSFMRequiredUnavailable,
    TSAgentKitError,
)

# ---- WS-3.1: fix_hint field + to_agent_dict ---------------------------------


class TestFixHintField:
    """The base error and subclasses carry an actionable fix_hint."""

    def test_base_error_default_fix_hint_empty(self) -> None:
        err = TSAgentKitError("something broke")
        assert err.fix_hint == ""

    def test_base_error_custom_fix_hint(self) -> None:
        err = TSAgentKitError("oops", fix_hint="try again")
        assert err.fix_hint == "try again"

    def test_fix_hint_override_in_constructor(self) -> None:
        """Instance fix_hint overrides the class-level default."""
        err = EDSNotMonotonic("bad data", fix_hint="custom hint")
        assert err.fix_hint == "custom hint"

    def test_fix_hint_none_preserves_class_default(self) -> None:
        """Passing fix_hint=None keeps the class-level default."""
        err = EDSNotMonotonic("bad data")
        assert "sort" in err.fix_hint.lower()

    def test_fix_hint_in_str_representation(self) -> None:
        err = TSAgentKitError("broken", fix_hint="restart")
        s = str(err)
        assert "[hint: restart]" in s

    def test_str_without_fix_hint_no_hint_tag(self) -> None:
        err = TSAgentKitError("broken")
        s = str(err)
        assert "[hint:" not in s

    def test_str_with_context_and_fix_hint(self) -> None:
        err = TSAgentKitError("broken", context={"k": "v"}, fix_hint="fix it")
        s = str(err)
        assert "(context:" in s
        assert "[hint: fix it]" in s


class TestToAgentDict:
    """to_agent_dict returns a well-structured dict."""

    def test_keys(self) -> None:
        err = TSAgentKitError("msg")
        d = err.to_agent_dict()
        assert set(d.keys()) == {"error_code", "message", "fix_hint", "context"}

    def test_values_default(self) -> None:
        err = TSAgentKitError("msg")
        d = err.to_agent_dict()
        assert d["error_code"] == "E_UNKNOWN"
        assert d["message"] == "msg"
        assert d["fix_hint"] == ""
        assert d["context"] == {}

    def test_values_with_all_fields(self) -> None:
        err = EContractMissingColumn("missing y", context={"col": "y"}, fix_hint="add col y")
        d = err.to_agent_dict()
        assert d["error_code"] == "E_CONTRACT_MISSING_COLUMN"
        assert d["message"] == "missing y"
        assert d["fix_hint"] == "add col y"
        assert d["context"] == {"col": "y"}

    def test_subclass_class_hint_appears(self) -> None:
        err = EDSNotMonotonic("unsorted")
        d = err.to_agent_dict()
        assert d["fix_hint"] != ""
        assert "sort" in d["fix_hint"].lower()


# ---- WS-3.2: Default fix_hints on high-frequency error classes ---------------


class TestDefaultFixHints:
    """Each high-frequency error class has a non-empty class-level fix_hint."""

    @pytest.mark.parametrize(
        "cls,keyword",
        [
            (EDSNotMonotonic, "sort"),
            (EContractMissingColumn, "columns"),
            (EContractDuplicateKey, "drop_duplicates"),
            (ECovariateLeakage, "past"),
            (ETSFMRequiredUnavailable, "install"),
            (EFallbackExhausted, "observations"),
            (EQAMinHistory, "historical"),
            (EBacktestInsufficientData, "reduce"),
            (EFreqInferFail, "freq"),
            (ECalibrationFail, "cross-validation"),
        ],
    )
    def test_class_fix_hint_non_empty_and_relevant(
        self,
        cls: type[TSAgentKitError],
        keyword: str,
    ) -> None:
        assert cls.fix_hint, f"{cls.__name__}.fix_hint should be non-empty"
        assert keyword.lower() in cls.fix_hint.lower(), (
            f"{cls.__name__}.fix_hint should contain '{keyword}'"
        )

    def test_base_error_has_empty_default(self) -> None:
        assert TSAgentKitError.fix_hint == ""

    def test_backward_compat_two_arg_construction(self) -> None:
        """Existing code using (message, context) still works."""
        err = EDSNotMonotonic("bad", {"col": "ds"})
        assert err.message == "bad"
        assert err.context == {"col": "ds"}
        assert "sort" in err.fix_hint.lower()
