"""TSFM adapters for tsagentkit.

Minimal wrappers for Time-Series Foundation Models:
- Chronos (Amazon)
- TimesFM (Google)
- Moirai (Salesforce)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Adapters are loaded on demand to avoid heavy imports at module load time
__all__ = [
    "ChronosAdapter",
    "TimesFMAdapter",
    "MoiraiAdapter",
]


def __getattr__(name: str):
    """Lazy load adapters to minimize import overhead."""
    if name == "ChronosAdapter":
        from tsagentkit.models.adapters.chronos import ChronosAdapter
        return ChronosAdapter
    elif name == "TimesFMAdapter":
        from tsagentkit.models.adapters.timesfm import TimesFMAdapter
        return TimesFMAdapter
    elif name == "MoiraiAdapter":
        from tsagentkit.models.adapters.moirai import MoiraiAdapter
        return MoiraiAdapter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
