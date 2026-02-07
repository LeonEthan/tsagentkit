"""Router module for tsagentkit.

Provides model selection and fallback strategies.
"""

from tsagentkit.contracts import RouteDecision

from .bucketing import (
    BucketConfig,
    BucketProfile,
    BucketStatistics,
    DataBucketer,
    SeriesBucket,
)
from .fallback import FallbackLadder, execute_with_fallback
from .plan import (
    PlanSpec,
    attach_plan_graph,
    build_plan_graph,
    compute_plan_signature,
    get_candidate_models,
)
from .router import get_model_for_series, inspect_tsfm_adapters, make_plan

__all__ = [
    # Plan
    "PlanSpec",
    "build_plan_graph",
    "attach_plan_graph",
    "compute_plan_signature",
    "get_candidate_models",
    # Router
    "make_plan",
    "get_model_for_series",
    "inspect_tsfm_adapters",
    "RouteDecision",
    # Fallback
    "FallbackLadder",
    "execute_with_fallback",
    # Bucketing (v0.2)
    "DataBucketer",
    "BucketConfig",
    "BucketProfile",
    "BucketStatistics",
    "SeriesBucket",
]
