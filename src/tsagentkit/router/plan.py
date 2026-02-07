"""PlanSpec helpers for routing and provenance."""

from __future__ import annotations

import hashlib
import json

from tsagentkit.contracts import PlanGraphSpec, PlanNodeSpec, PlanSpec


def compute_plan_signature(plan: PlanSpec) -> str:
    """Compute deterministic signature for a PlanSpec."""
    data = plan.model_dump()
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def get_candidate_models(plan: PlanSpec) -> list[str]:
    """Return ordered candidate models for a plan."""
    return list(plan.candidate_models)


def build_plan_graph(
    plan: PlanSpec,
    include_backtest: bool = False,
) -> PlanGraphSpec:
    """Build a deterministic execution graph from a PlanSpec.

    The graph is assembly-oriented: it exposes reusable execution nodes and the
    ordered candidate-model attempt chain used for fallback.
    """
    nodes: list[PlanNodeSpec] = []
    nodes.append(PlanNodeSpec(node_id="validate", kind="validate"))
    nodes.append(PlanNodeSpec(node_id="qa", kind="qa", depends_on=["validate"]))
    nodes.append(PlanNodeSpec(node_id="align_covariates", kind="align_covariates", depends_on=["qa"]))
    nodes.append(PlanNodeSpec(node_id="build_dataset", kind="build_dataset", depends_on=["align_covariates"]))
    nodes.append(PlanNodeSpec(node_id="make_plan", kind="make_plan", depends_on=["build_dataset"]))

    previous = "make_plan"
    if include_backtest:
        nodes.append(PlanNodeSpec(node_id="backtest", kind="backtest", depends_on=[previous]))
        previous = "backtest"

    for idx, model_name in enumerate(get_candidate_models(plan)):
        fit_id = f"fit_{idx}"
        predict_id = f"predict_{idx}"
        nodes.append(
            PlanNodeSpec(
                node_id=fit_id,
                kind="fit",
                depends_on=[previous],
                model_name=model_name,
                group="model_attempts",
                metadata={"attempt_index": idx},
            )
        )
        nodes.append(
            PlanNodeSpec(
                node_id=predict_id,
                kind="predict",
                depends_on=[fit_id],
                model_name=model_name,
                group="model_attempts",
                metadata={"attempt_index": idx},
            )
        )
        previous = predict_id

    nodes.append(PlanNodeSpec(node_id="package", kind="package", depends_on=[previous]))

    return PlanGraphSpec(
        plan_name=plan.plan_name,
        nodes=nodes,
        entrypoints=["validate"],
        terminal_nodes=["package"],
    )


def attach_plan_graph(plan: PlanSpec, include_backtest: bool = False) -> PlanSpec:
    """Return a new plan with graph metadata attached."""
    graph = build_plan_graph(plan, include_backtest=include_backtest)
    return plan.model_copy(update={"graph": graph})


__all__ = [
    "PlanSpec",
    "compute_plan_signature",
    "get_candidate_models",
    "build_plan_graph",
    "attach_plan_graph",
]
