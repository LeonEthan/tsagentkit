"""Tests for PRD-aligned router."""

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.contracts import ETSFMRequiredUnavailable
from tsagentkit.router import get_model_for_series, make_plan
from tsagentkit.series import TSDataset


def _make_dataset(values: list[float]) -> TSDataset:
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * len(values),
            "ds": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "y": values,
        }
    )
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})
    return TSDataset.from_dataframe(df, spec)


def test_make_plan_regular_series() -> None:
    dataset = _make_dataset([1.0] * 30)
    plan, route_decision = make_plan(dataset, dataset.task_spec)
    assert plan.candidate_models[0] in {"HistoricAverage", "tsfm-chronos", "tsfm-moirai", "tsfm-timesfm"}
    # Verify RouteDecision is returned correctly
    assert route_decision.selected_plan == plan
    assert "reasons" in route_decision.model_dump()


def test_make_plan_intermittent_series() -> None:
    dataset = _make_dataset([0.0] * 10 + [1.0] + [0.0] * 14 + [10.0] + [0.0] * 14 + [2.0])
    plan, route_decision = make_plan(dataset, dataset.task_spec, use_tsfm=False)
    assert plan.candidate_models[0] == "Croston"
    # Verify RouteDecision contains bucket info
    assert "intermittent" in route_decision.buckets


def test_get_model_for_series_short_history() -> None:
    dataset = _make_dataset([1.0] * 5)
    model = get_model_for_series("A", dataset, dataset.task_spec)
    assert model == "HistoricAverage"


def test_make_plan_tsfm_required_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset([1.0] * 30)
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "required"})

    monkeypatch.setattr(
        "tsagentkit.models.adapters.AdapterRegistry.check_availability",
        lambda name: (False, f"{name} missing"),
    )

    with pytest.raises(ETSFMRequiredUnavailable):
        make_plan(dataset, spec)


def test_make_plan_tsfm_required_uses_tsfm_only(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset([1.0] * 30)
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "required"})

    monkeypatch.setattr(
        "tsagentkit.models.adapters.AdapterRegistry.check_availability",
        lambda name: (name == "chronos", "" if name == "chronos" else "missing"),
    )

    plan, _route_decision = make_plan(dataset, spec, tsfm_preference=["chronos"])
    assert plan.candidate_models == ["tsfm-chronos"]
    assert plan.allow_baseline is False


def test_make_plan_tsfm_disabled_ignores_available(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset([1.0] * 30)
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "disabled"})

    monkeypatch.setattr(
        "tsagentkit.models.adapters.AdapterRegistry.check_availability",
        lambda name: (True, ""),
    )

    plan, _route_decision = make_plan(dataset, spec)
    assert all(not model.startswith("tsfm-") for model in plan.candidate_models)


def test_make_plan_disallow_non_tsfm_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset([1.0] * 30)
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

    with pytest.raises(ETSFMRequiredUnavailable):
        make_plan(dataset, spec)
