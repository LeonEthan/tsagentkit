"""Tests for router PlanSpec helpers."""

import pytest
from pydantic import ValidationError

from tsagentkit.router import PlanSpec, compute_plan_signature


class TestPlanSpec:
    def test_minimal_plan(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])
        assert plan.plan_name == "default"
        assert plan.candidate_models == ["SeasonalNaive"]

    def test_signature_deterministic(self) -> None:
        plan1 = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])
        plan2 = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])
        assert compute_plan_signature(plan1) == compute_plan_signature(plan2)

    def test_signature_unique(self) -> None:
        plan1 = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])
        plan2 = PlanSpec(plan_name="default", candidate_models=["Naive"])
        assert compute_plan_signature(plan1) != compute_plan_signature(plan2)

    def test_frozen(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])
        with pytest.raises(ValidationError):
            plan.plan_name = "other"  # type: ignore
