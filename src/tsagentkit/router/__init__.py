"""Router module for tsagentkit.

Provides model selection and fallback strategies.
"""

from .fallback import FallbackLadder, execute_with_fallback
from .plan import Plan
from .router import get_model_for_series, make_plan

__all__ = [
    # Plan
    "Plan",
    # Router
    "make_plan",
    "get_model_for_series",
    # Fallback
    "FallbackLadder",
    "execute_with_fallback",
]
