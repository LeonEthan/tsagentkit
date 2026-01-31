"""Tests for router/fallback.py."""

from unittest.mock import MagicMock

import pytest

from tsagentkit.contracts import EFallbackExhausted, EModelFitFailed
from tsagentkit.router import FallbackLadder, execute_with_fallback
from tsagentkit.router.plan import Plan


class TestExecuteWithFallback:
    """Tests for execute_with_fallback function."""

    def test_primary_succeeds(self) -> None:
        """Test when primary model succeeds."""
        plan = Plan(
            primary_model="ModelA",
            fallback_chain=["ModelB", "ModelC"],
        )

        def fit_func(model: str, dataset: str, config: dict) -> str:
            return f"result_{model}"

        result, model_name = execute_with_fallback(fit_func, "dataset", plan)
        assert result == "result_ModelA"
        assert model_name == "ModelA"

    def test_fallback_triggered(self) -> None:
        """Test when fallback is triggered."""
        plan = Plan(
            primary_model="ModelA",
            fallback_chain=["ModelB", "ModelC"],
        )

        call_count = 0

        def fit_func(model: str, dataset: str, config: dict) -> str:
            nonlocal call_count
            call_count += 1
            if model == "ModelA":
                raise EModelFitFailed("Model A failed")
            return f"result_{model}"

        result, model_name = execute_with_fallback(fit_func, "dataset", plan)
        assert result == "result_ModelB"
        assert model_name == "ModelB"
        assert call_count == 2

    def test_all_models_fail(self) -> None:
        """Test when all models fail."""
        plan = Plan(
            primary_model="ModelA",
            fallback_chain=["ModelB"],
        )

        def fit_func(model: str, dataset: str, config: dict) -> str:
            raise EModelFitFailed(f"{model} failed")

        with pytest.raises(EFallbackExhausted) as exc_info:
            execute_with_fallback(fit_func, "dataset", plan)

        assert "ModelA" in str(exc_info.value)
        assert "ModelB" in str(exc_info.value)

    def test_callback_triggered(self) -> None:
        """Test callback is triggered on fallback."""
        plan = Plan(
            primary_model="ModelA",
            fallback_chain=["ModelB"],
        )

        def fit_func(model: str, dataset: str, config: dict) -> str:
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

    def test_no_callback_on_success(self) -> None:
        """Test callback not called when primary succeeds."""
        plan = Plan(primary_model="ModelA")

        def fit_func(model: str, dataset: str, config: dict) -> str:
            return "success"

        callback_calls = []

        def on_fallback(from_model: str, to_model: str, error: Exception) -> None:
            callback_calls.append((from_model, to_model))

        execute_with_fallback(fit_func, "dataset", plan, on_fallback=on_fallback)

        assert len(callback_calls) == 0


class TestFallbackLadder:
    """Tests for FallbackLadder class."""

    def test_standard_ladder(self) -> None:
        """Test standard fallback ladder."""
        ladder = FallbackLadder.STANDARD_LADDER
        assert ladder == ["SeasonalNaive", "HistoricAverage", "Naive"]

    def test_intermittent_ladder(self) -> None:
        """Test intermittent demand ladder."""
        ladder = FallbackLadder.INTERMITTENT_LADDER
        assert ladder == ["Croston", "Naive"]

    def test_cold_start_ladder(self) -> None:
        """Test cold-start ladder."""
        ladder = FallbackLadder.COLD_START_LADDER
        assert ladder == ["HistoricAverage", "Naive"]

    def test_get_ladder_standard(self) -> None:
        """Test getting standard ladder."""
        ladder = FallbackLadder.get_ladder(is_intermittent=False, is_cold_start=False)
        assert ladder == FallbackLadder.STANDARD_LADDER

    def test_get_ladder_intermittent(self) -> None:
        """Test getting intermittent ladder."""
        ladder = FallbackLadder.get_ladder(is_intermittent=True, is_cold_start=False)
        assert ladder == FallbackLadder.INTERMITTENT_LADDER

    def test_get_ladder_cold_start(self) -> None:
        """Test getting cold-start ladder."""
        ladder = FallbackLadder.get_ladder(is_intermittent=False, is_cold_start=True)
        assert ladder == FallbackLadder.COLD_START_LADDER

    def test_with_primary_standard(self) -> None:
        """Test creating chain with primary model."""
        chain = FallbackLadder.with_primary("TSFM", is_intermittent=False)
        assert chain[0] == "TSFM"
        assert "SeasonalNaive" in chain

    def test_with_primary_intermittent(self) -> None:
        """Test creating chain for intermittent."""
        chain = FallbackLadder.with_primary("Croston", is_intermittent=True)
        assert chain[0] == "Croston"
        assert "Naive" in chain

    def test_with_primary_no_duplicate(self) -> None:
        """Test primary is not duplicated in fallbacks."""
        chain = FallbackLadder.with_primary(
            "SeasonalNaive",
            fallbacks=["SeasonalNaive", "Naive"],
        )
        assert chain.count("SeasonalNaive") == 1
        assert chain == ["SeasonalNaive", "Naive"]

    def test_with_primary_custom_fallbacks(self) -> None:
        """Test using custom fallback list."""
        custom_fallbacks = ["CustomModel", "Naive"]
        chain = FallbackLadder.with_primary(
            "PrimaryModel",
            fallbacks=custom_fallbacks,
        )
        assert chain == ["PrimaryModel", "CustomModel", "Naive"]
