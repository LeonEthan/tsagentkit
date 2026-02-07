"""Tests for router/fallback.py."""

import pytest

from tsagentkit.contracts import EFallbackExhausted, EModelFitFailed
from tsagentkit.router import FallbackLadder, PlanSpec, execute_with_fallback


class TestExecuteWithFallback:
    def test_primary_succeeds(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["ModelA", "ModelB"])

        def fit_func(model: str, dataset: str) -> str:
            return f"result_{model}"

        result, model_name = execute_with_fallback(fit_func, "dataset", plan)
        assert result == "result_ModelA"
        assert model_name == "ModelA"

    def test_fallback_triggered(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["ModelA", "ModelB"])

        def fit_func(model: str, dataset: str) -> str:
            if model == "ModelA":
                raise EModelFitFailed("Model A failed")
            return f"result_{model}"

        result, model_name = execute_with_fallback(fit_func, "dataset", plan)
        assert result == "result_ModelB"
        assert model_name == "ModelB"

    def test_all_models_fail(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["ModelA", "ModelB"])

        def fit_func(model: str, dataset: str) -> str:
            raise EModelFitFailed(f"{model} failed")

        with pytest.raises(EFallbackExhausted):
            execute_with_fallback(fit_func, "dataset", plan)

    def test_callback_triggered(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["ModelA", "ModelB"])

        def fit_func(model: str, dataset: str) -> str:
            if model == "ModelA":
                raise EModelFitFailed("Model A failed")
            return "success"

        callback_calls = []

        def on_fallback(from_model: str, to_model: str, error: Exception) -> None:
            callback_calls.append((from_model, to_model, str(error)))

        execute_with_fallback(fit_func, "dataset", plan, on_fallback=on_fallback)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "ModelA"
        assert callback_calls[0][1] == "ModelB"

    def test_non_fallback_error_raises_immediately(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["ModelA", "ModelB"])

        def fit_func(model: str, dataset: str) -> str:
            if model == "ModelA":
                raise RuntimeError("unexpected crash")
            return "success"

        with pytest.raises(RuntimeError, match="unexpected crash"):
            execute_with_fallback(fit_func, "dataset", plan)


class TestFallbackLadder:
    def test_standard_ladder(self) -> None:
        assert FallbackLadder.STANDARD_LADDER == ["SeasonalNaive", "HistoricAverage", "Naive"]

    def test_intermittent_ladder(self) -> None:
        assert FallbackLadder.INTERMITTENT_LADDER == ["Croston", "Naive"]

    def test_cold_start_ladder(self) -> None:
        assert FallbackLadder.COLD_START_LADDER == ["HistoricAverage", "Naive"]
