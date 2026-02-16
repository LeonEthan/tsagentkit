"""tsagentkit - Minimalist time-series forecasting for AI agents.

This library provides a strict, production-grade execution engine for
time-series forecasting with TSFM-first strategy and automatic fallback.

Version 2.0.0 - Nanobot-Inspired Refactoring

Basic usage:
    >>> from tsagentkit import forecast
    >>> result = forecast(data, h=7)
    >>> print(result.df)

Advanced usage:
    >>> from tsagentkit import ForecastConfig, run_forecast
    >>> config = ForecastConfig.quick(h=14, freq='H')
    >>> result = run_forecast(data, config)

Agent Building (granular control):
    >>> from tsagentkit import validate, TSDataset, ModelCache
    >>> from tsagentkit.models.registry import REGISTRY, list_available
    >>> df = validate(data)
    >>> dataset = TSDataset.from_dataframe(df, config)
    >>> models = [REGISTRY[m] for m in list_available(tsfm_only=True)]
    >>> ModelCache.preload(models)  # Load all TSFMs once
    >>> for batch in batches:
    ...     result = forecast(batch, h=7)  # Uses cached models
    >>> ModelCache.unload()  # Free memory when done
"""

__version__ = "2.0.0"

# Core API
from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.data import CovariateSet, TSDataset
from tsagentkit.core.errors import (
    EContract,
    EInsufficient,
    ENoTSFM,
    ETemporal,
    TSAgentKitError,
)
from tsagentkit.core.results import ForecastResult, RunResult

# Main entry points
from tsagentkit.pipeline.runner import forecast, run_pipeline

# Model Cache (for explicit lifecycle management)
from tsagentkit.models.cache import ModelCache

# Registry (for agent building)
from tsagentkit.models.registry import REGISTRY, ModelSpec, list_models, list_available

# Inspection utilities
from tsagentkit.inspect import check_health, list_models as inspect_list_models

# Legacy aliases for backward compatibility
EContractViolation = EContract
EDataQuality = EContract
EModelFailed = EInsufficient
ETSFMRequired = ENoTSFM

__all__ = [
    "__version__",
    # Core
    "forecast",
    "run_pipeline",
    "ForecastConfig",
    "TSDataset",
    "CovariateSet",
    "RunResult",
    "ForecastResult",
    # Model Cache
    "ModelCache",
    # Registry
    "REGISTRY",
    "ModelSpec",
    "list_models",
    "list_available",
    # Inspection
    "check_health",
    "inspect_list_models",
    # Errors (new names)
    "TSAgentKitError",
    "EContract",
    "ENoTSFM",
    "EInsufficient",
    "ETemporal",
    # Errors (legacy aliases)
    "EContractViolation",
    "EDataQuality",
    "EModelFailed",
    "ETSFMRequired",
]
