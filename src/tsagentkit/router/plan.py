"""PlanSpec helpers for routing and provenance."""

from __future__ import annotations

import hashlib
import json

from tsagentkit.contracts import PlanSpec


def compute_plan_signature(plan: PlanSpec) -> str:
    """Compute deterministic signature for a PlanSpec."""
    data = plan.model_dump()
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def get_candidate_models(plan: PlanSpec) -> list[str]:
    """Return ordered candidate models for a plan."""
    return list(plan.candidate_models)


__all__ = ["PlanSpec", "compute_plan_signature", "get_candidate_models"]
