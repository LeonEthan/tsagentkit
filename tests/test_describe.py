"""Tests for tsagentkit.describe() API discovery function."""

from __future__ import annotations

from tsagentkit.discovery import describe


def test_describe_returns_dict() -> None:
    """describe() returns a dictionary."""
    result = describe()
    assert isinstance(result, dict)


def test_describe_has_expected_top_level_keys() -> None:
    """Result has version, apis, error_codes, tsfm_adapters."""
    result = describe()
    expected_keys = {"version", "apis", "error_codes", "tsfm_adapters"}
    assert expected_keys.issubset(result.keys())


def test_describe_version_matches_package() -> None:
    """version field matches tsagentkit.__version__."""
    import tsagentkit

    result = describe()
    assert result["version"] == tsagentkit.__version__


def test_describe_error_codes_contains_registry() -> None:
    """error_codes contains all entries from ERROR_REGISTRY."""
    from tsagentkit.contracts.errors import ERROR_REGISTRY

    result = describe()
    error_codes = result["error_codes"]

    for code in ERROR_REGISTRY:
        assert code in error_codes, f"Missing error code: {code}"
        assert "class" in error_codes[code]
        assert "description" in error_codes[code]
        assert "fix_hint" in error_codes[code]


def test_describe_error_codes_fix_hints() -> None:
    """Error codes with class-level fix_hints are correctly populated."""
    result = describe()
    error_codes = result["error_codes"]

    # Spot-check a few codes that have fix_hints
    assert error_codes["E_DS_NOT_MONOTONIC"]["fix_hint"] != ""
    assert "sort" in error_codes["E_DS_NOT_MONOTONIC"]["fix_hint"].lower()

    assert error_codes["E_CONTRACT_MISSING_COLUMN"]["fix_hint"] != ""
    assert "unique_id" in error_codes["E_CONTRACT_MISSING_COLUMN"]["fix_hint"]


def test_describe_tsfm_adapters_is_list() -> None:
    """tsfm_adapters is a list."""
    result = describe()
    assert isinstance(result["tsfm_adapters"], list)


def test_describe_tsfm_adapter_entries_have_expected_keys() -> None:
    """Each adapter entry has name, registered, available, reason keys."""
    result = describe()
    for adapter in result["tsfm_adapters"]:
        assert "name" in adapter
        assert "registered" in adapter
        assert "available" in adapter
        assert "reason" in adapter


def test_describe_apis_is_dict() -> None:
    """apis field is a dictionary with entries."""
    result = describe()
    apis = result["apis"]
    assert isinstance(apis, dict)
    assert len(apis) > 0


def test_describe_apis_contain_core_functions() -> None:
    """apis includes the core assembly functions."""
    result = describe()
    apis = result["apis"]

    # Check for key pipeline steps
    core_tasks = ["validate", "qa", "build_dataset", "make_plan", "fit", "predict", "package"]
    for task in core_tasks:
        assert task in apis, f"Missing API task: {task}"
        assert "function" in apis[task]
        assert "description" in apis[task]
