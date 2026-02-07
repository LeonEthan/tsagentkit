"""Tests for router PlanSpec helpers."""

import pytest
from pydantic import ValidationError

from tsagentkit.router import (
    PlanSpec,
    attach_plan_graph,
    build_plan_graph,
    compute_plan_signature,
)


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

    def test_build_plan_graph(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["ModelA", "ModelB"])
        graph = build_plan_graph(plan)

        assert graph.plan_name == "default"
        assert graph.entrypoints == ["validate"]
        assert graph.terminal_nodes == ["package"]
        assert [node.node_id for node in graph.nodes[:3]] == ["validate", "qa", "align_covariates"]
        assert any(node.model_name == "ModelA" for node in graph.nodes)
        assert any(node.model_name == "ModelB" for node in graph.nodes)

    def test_attach_plan_graph(self) -> None:
        plan = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])
        enriched = attach_plan_graph(plan, include_backtest=True)

        assert enriched.graph is not None
        assert any(node.node_id == "backtest" for node in enriched.graph.nodes)
