"""Deterministic TSFM policy matrix checks for CI.

These tests intentionally monkeypatch adapter availability so CI can assert
policy behavior without depending on host-specific package/runtime state.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.contracts import ETSFMRequiredUnavailable
from tsagentkit.router import make_plan
from tsagentkit.series import TSDataset


def _dataset() -> TSDataset:
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
            "y": [float(v) for v in range(30)],
        }
    )
    return TSDataset.from_dataframe(df, TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"}))


def test_matrix_required_mode_unavailable_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset()
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "required"})

    monkeypatch.setattr(
        "tsagentkit.models.adapters.AdapterRegistry.check_availability",
        lambda name: (False, f"{name} missing"),
    )

    with pytest.raises(ETSFMRequiredUnavailable) as exc:
        make_plan(dataset, spec)

    assert exc.value.error_code == "E_TSFM_REQUIRED_UNAVAILABLE"


def test_matrix_required_mode_available_uses_tsfm_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset()
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "required"})

    monkeypatch.setattr(
        "tsagentkit.models.adapters.AdapterRegistry.check_availability",
        lambda name: (name == "chronos", "" if name == "chronos" else f"{name} missing"),
    )

    plan, _decision = make_plan(dataset, spec, tsfm_preference=["chronos", "moirai", "timesfm"])
    assert plan.candidate_models == ["tsfm-chronos"]
    assert plan.allow_baseline is False


def test_matrix_preferred_disallow_non_tsfm_fallback_unavailable_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    spec = TaskSpec(
        h=7,
        freq="D",
        tsfm_policy={
            "mode": "preferred",
            "allow_non_tsfm_fallback": False,
        },
    )

    monkeypatch.setattr(
        "tsagentkit.models.adapters.AdapterRegistry.check_availability",
        lambda name: (False, f"{name} missing"),
    )

    with pytest.raises(ETSFMRequiredUnavailable) as exc:
        make_plan(dataset, spec)

    assert exc.value.error_code == "E_TSFM_REQUIRED_UNAVAILABLE"
