"""Models module for tsagentkit.

Minimal model fitting and prediction using the registry/cache/protocol pattern.
"""

from tsagentkit.models.cache import ModelCache
from tsagentkit.models.ensemble import ensemble, ensemble_with_quantiles
from tsagentkit.models.protocol import fit, predict
from tsagentkit.models.registry import (
    REGISTRY,
    ModelSpec,
    get_spec,
    list_available,
    list_models,
)

__all__ = [
    # Registry
    "REGISTRY",
    "ModelSpec",
    "list_models",
    "get_spec",
    "list_available",
    # Cache
    "ModelCache",
    # Protocol
    "fit",
    "predict",
    # Ensemble
    "ensemble",
    "ensemble_with_quantiles",
]
