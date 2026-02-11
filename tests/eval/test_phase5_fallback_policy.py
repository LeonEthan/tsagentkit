"""Phase 5 tests for preload policy and fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.contracts import EFallbackExhausted, EModelNotLoaded, EModelPredictFailed, PlanSpec
from tsagentkit.gift_eval import predictor as predictor_mod
from tsagentkit.serving import orchestration as orch


class _DatasetStub:
    def __init__(self, horizon: int = 2) -> None:
        self.task_spec = SimpleNamespace(horizon=horizon, season_length=1)
        self.hierarchy = None

    def is_hierarchical(self) -> bool:
        return False


class _FakeAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.fit_calls = 0

    def fit(self, dataset, prediction_length: int, quantiles):  # noqa: ANN001
        _ = dataset
        _ = prediction_length
        _ = quantiles
        self.fit_calls += 1


class _FakePool:
    def __init__(self, adapters: dict[str, _FakeAdapter]) -> None:
        self.adapters = adapters
        self.requested: list[str] = []

    def get(self, adapter_name: str) -> _FakeAdapter:
        self.requested.append(adapter_name)
        if adapter_name not in self.adapters:
            raise EModelNotLoaded(
                f"Adapter '{adapter_name}' is configured but not preloaded in this session."
            )
        return self.adapters[adapter_name]


def _make_predictor_without_init(monkeypatch) -> predictor_mod.TSAgentKitPredictor:
    monkeypatch.setattr(predictor_mod.TSAgentKitPredictor, "_init_session", lambda self: None)
    predictor = predictor_mod.TSAgentKitPredictor(mode="quick", batch_size=1)
    predictor._session = "phase5-session"
    return predictor


def _predict_with_first_failure(dataset, artifact, spec, covariates=None):  # noqa: ANN001
    _ = dataset
    _ = spec
    _ = covariates
    if artifact.model_name == "tsfm-moirai":
        return pd.DataFrame(
            {
                "unique_id": ["uid-1", "uid-1"],
                "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
                "yhat": [1.0, 2.0],
                "model": ["tsfm-moirai", "tsfm-moirai"],
            }
        )
    raise EModelPredictFailed("simulated predict failure for primary adapter")


def test_phase5_fallback_uses_next_preloaded_adapter(monkeypatch) -> None:
    predictor = _make_predictor_without_init(monkeypatch)
    pool = _FakePool(
        {
            "chronos": _FakeAdapter("chronos"),
            "moirai": _FakeAdapter("moirai"),
        }
    )
    predictor._model_pool = pool

    dataset = _DatasetStub(horizon=2)
    task_spec = TaskSpec(h=2, freq="D")
    plan = PlanSpec(
        plan_name="phase5-fallback",
        candidate_models=["tsfm-chronos", "tsfm-moirai"],
        quantiles=[0.1, 0.5],
    )
    transitions: list[tuple[str, str]] = []

    artifact, forecast = orch._fit_predict_with_fallback(
        dataset=dataset,
        plan=plan,
        task_spec=task_spec,
        fit_func=predictor._fit_with_model_pool,
        predict_func=_predict_with_first_failure,
        start_after="tsfm-chronos",
        initial_error=EModelPredictFailed("chronos predict failed"),
        on_fallback=lambda frm, to, err: transitions.append((frm, to)),
    )

    assert artifact.model_name == "tsfm-moirai"
    assert not forecast.empty
    assert transitions == [("tsfm-chronos", "tsfm-moirai")]
    assert pool.requested == ["moirai"]


def test_phase5_guardrail_raises_when_non_preloaded_adapter_selected(monkeypatch) -> None:
    predictor = _make_predictor_without_init(monkeypatch)
    pool = _FakePool({"chronos": _FakeAdapter("chronos")})
    predictor._model_pool = pool

    dataset = _DatasetStub(horizon=2)
    task_spec = TaskSpec(h=2, freq="D")
    plan = PlanSpec(
        plan_name="phase5-guardrail",
        candidate_models=["tsfm-chronos", "tsfm-moirai"],
        quantiles=[0.1, 0.5],
    )

    with pytest.raises(EFallbackExhausted, match="not preloaded"):
        orch._fit_predict_with_fallback(
            dataset=dataset,
            plan=plan,
            task_spec=task_spec,
            fit_func=predictor._fit_with_model_pool,
            predict_func=_predict_with_first_failure,
            start_after="tsfm-chronos",
            initial_error=EModelPredictFailed("chronos predict failed"),
            on_fallback=None,
        )
