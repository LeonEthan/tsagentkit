"""TSFM (Time-Series Foundation Model) adapters for tsagentkit.

This module provides unified adapters for popular time-series foundation models:
- Amazon Chronos
- Salesforce Moirai
- Google TimesFM

Example:
    >>> from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry
    >>> from tsagentkit.models.adapters import ChronosAdapter
    >>>
    >>> # Register adapters
    >>> AdapterRegistry.register("chronos", ChronosAdapter)
    >>>
    >>> # Create and use adapter
    >>> config = AdapterConfig(model_name="chronos", model_size="base")
    >>> adapter = AdapterRegistry.create("chronos", config)
    >>> adapter.load_model()
    >>> result = adapter.predict(dataset, horizon=30)
"""

from __future__ import annotations

from typing import TypeVar

# Base classes
from .base import AdapterConfig, TSFMAdapter
from .registry import AdapterRegistry

__all__ = [
    # Base classes
    "AdapterConfig",
    "TSFMAdapter",
    "AdapterRegistry",
]

# Type variable for adapter classes
T = TypeVar("T", bound=type[TSFMAdapter])

# Lazily import specific adapters as they become available
# This prevents import errors when dependencies are not installed


def _try_import_chronos() -> type[TSFMAdapter] | None:
    """Try to import ChronosAdapter if dependencies are available."""
    try:
        from .chronos import ChronosAdapter

        return ChronosAdapter
    except ImportError:
        return None


def _try_import_moirai() -> type[TSFMAdapter] | None:
    """Try to import MoiraiAdapter if dependencies are available."""
    try:
        from .moirai import MoiraiAdapter

        return MoiraiAdapter
    except ImportError:
        return None


def _try_import_timesfm() -> type[TSFMAdapter] | None:
    """Try to import TimesFMAdapter if dependencies are available."""
    try:
        from .timesfm import TimesFMAdapter

        return TimesFMAdapter
    except ImportError:
        return None


# Auto-register available adapters
_chronos = _try_import_chronos()
if _chronos:
    AdapterRegistry.register("chronos", _chronos)
    __all__.append("ChronosAdapter")

_moirai = _try_import_moirai()
if _moirai:
    AdapterRegistry.register("moirai", _moirai)
    __all__.append("MoiraiAdapter")

_timesfm = _try_import_timesfm()
if _timesfm:
    AdapterRegistry.register("timesfm", _timesfm)
    __all__.append("TimesFMAdapter")
