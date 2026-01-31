"""Tests for router/plan.py."""

import pytest

from tsagentkit.router import Plan


class TestPlanCreation:
    """Tests for Plan creation."""

    def test_minimal_plan(self) -> None:
        """Test creating a minimal plan."""
        plan = Plan(primary_model="SeasonalNaive")
        assert plan.primary_model == "SeasonalNaive"
        assert plan.fallback_chain == []
        assert plan.signature != ""

    def test_plan_with_fallbacks(self) -> None:
        """Test creating plan with fallback chain."""
        plan = Plan(
            primary_model="SeasonalNaive",
            fallback_chain=["HistoricAverage", "Naive"],
            config={"season_length": 7},
        )
        assert plan.primary_model == "SeasonalNaive"
        assert plan.fallback_chain == ["HistoricAverage", "Naive"]
        assert plan.config == {"season_length": 7}

    def test_plan_with_strategy(self) -> None:
        """Test creating plan with strategy."""
        plan = Plan(
            primary_model="SeasonalNaive",
            strategy="baseline_only",
        )
        assert plan.strategy == "baseline_only"

    def test_signature_auto_computed(self) -> None:
        """Test that signature is auto-computed."""
        plan = Plan(primary_model="SeasonalNaive")
        assert len(plan.signature) == 16
        assert all(c in "0123456789abcdef" for c in plan.signature)

    def test_signature_deterministic(self) -> None:
        """Test that same plan produces same signature."""
        plan1 = Plan(
            primary_model="SeasonalNaive",
            fallback_chain=["Naive"],
            config={"season_length": 7},
        )
        plan2 = Plan(
            primary_model="SeasonalNaive",
            fallback_chain=["Naive"],
            config={"season_length": 7},
        )
        assert plan1.signature == plan2.signature

    def test_signature_unique(self) -> None:
        """Test that different plans have different signatures."""
        plan1 = Plan(primary_model="SeasonalNaive")
        plan2 = Plan(primary_model="HistoricAverage")
        assert plan1.signature != plan2.signature


class TestPlanMethods:
    """Tests for Plan methods."""

    def test_get_all_models_no_fallbacks(self) -> None:
        """Test getting all models with no fallbacks."""
        plan = Plan(primary_model="SeasonalNaive")
        assert plan.get_all_models() == ["SeasonalNaive"]

    def test_get_all_models_with_fallbacks(self) -> None:
        """Test getting all models with fallbacks."""
        plan = Plan(
            primary_model="SeasonalNaive",
            fallback_chain=["HistoricAverage", "Naive"],
        )
        assert plan.get_all_models() == ["SeasonalNaive", "HistoricAverage", "Naive"]

    def test_to_signature_no_fallbacks(self) -> None:
        """Test human-readable signature without fallbacks."""
        plan = Plan(primary_model="SeasonalNaive")
        assert plan.to_signature() == "Plan(SeasonalNaive)"

    def test_to_signature_with_fallbacks(self) -> None:
        """Test human-readable signature with fallbacks."""
        plan = Plan(
            primary_model="SeasonalNaive",
            fallback_chain=["HistoricAverage", "Naive"],
        )
        assert plan.to_signature() == "Plan(SeasonalNaive->HistoricAverage->Naive)"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        plan = Plan(
            primary_model="SeasonalNaive",
            fallback_chain=["Naive"],
            config={"season_length": 7},
            strategy="auto",
        )
        d = plan.to_dict()
        assert d["primary_model"] == "SeasonalNaive"
        assert d["fallback_chain"] == ["Naive"]
        assert d["config"] == {"season_length": 7}
        assert d["strategy"] == "auto"
        assert "signature" in d

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "primary_model": "SeasonalNaive",
            "fallback_chain": ["Naive"],
            "config": {"season_length": 7},
            "strategy": "auto",
            "signature": "abc123",
        }
        plan = Plan.from_dict(data)
        assert plan.primary_model == "SeasonalNaive"
        assert plan.fallback_chain == ["Naive"]
        assert plan.signature == "abc123"


class TestPlanImmutability:
    """Tests for Plan immutability."""

    def test_frozen_dataclass(self) -> None:
        """Test that Plan is frozen."""
        plan = Plan(primary_model="SeasonalNaive")
        with pytest.raises(Exception):
            plan.primary_model = "Naive"  # type: ignore

