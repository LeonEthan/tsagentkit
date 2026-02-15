"""Router module for tsagentkit.

Provides model selection and execution planning.
"""

from tsagentkit.router.plan import (
    ModelCandidate,
    Plan,
    build_plan,
    inspect_tsfm_adapters,
)

__all__ = [
    "Plan",
    "ModelCandidate",
    "build_plan",
    "inspect_tsfm_adapters",
]
