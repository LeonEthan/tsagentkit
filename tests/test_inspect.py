"""Tests for inspection helpers."""

from __future__ import annotations

import tsagentkit.inspect as inspect_module
from tsagentkit.models.registry import REGISTRY


def test_list_models_returns_registry_tsfms():
    """Inspect list_models should reflect registry TSFM entries."""
    expected = [name for name, spec in REGISTRY.items() if spec.is_tsfm]
    actual = inspect_module.list_models(tsfm_only=True)
    assert actual == expected


def test_check_health_uses_registry_for_tsfm_status():
    """Health report should expose registry TSFMs with no missing-dependency probing."""
    report = inspect_module.check_health()
    expected = [name for name, spec in REGISTRY.items() if spec.is_tsfm]
    assert report.tsfm_available == expected
    assert report.tsfm_missing == []
    assert report.all_ok is True
