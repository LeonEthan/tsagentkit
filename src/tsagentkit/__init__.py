"""tsagentkit - Minimalist time-series forecasting for AI agents.

Ultra-lightweight execution engine for time-series forecasting with TSFM ensemble.

Version 2.0.0 - Nanobot-Inspired Architecture

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
from tsagentkit.core.dataset import CovariateSet, TSDataset
from tsagentkit.core.errors import (
    EContract,
    EInsufficient,
    ENoTSFM,
    ETemporal,
    TSAgentKitError,
)
from tsagentkit.core.results import ForecastResult, RunResult

# Main entry points (Standard Pipeline)
from tsagentkit.pipeline import forecast, run_forecast

# Agent Building (granular control)
from tsagentkit.pipeline import (
    build_dataset,
    ensemble,
    fit_all,
    make_plan,
    predict_all,
    validate,
)

# Model Cache (for explicit lifecycle management)
from tsagentkit.models.cache import ModelCache

# Registry (for agent building)
from tsagentkit.models.registry import REGISTRY, ModelSpec, list_models

# Inspection utilities
from tsagentkit.inspect import check_health, list_models as inspect_list_models

__all__ = [
    "__version__",
    # Standard Pipeline
    "forecast",
    "run_forecast",
    "ForecastConfig",
    "ForecastResult",
    "TSDataset",
    "CovariateSet",
    # Agent Building
    "validate",
    "build_dataset",
    "make_plan",
    "fit_all",
    "predict_all",
    "ensemble",
    # Model Cache
    "ModelCache",
    # Registry
    "REGISTRY",
    "ModelSpec",
    "list_models",
    # Inspection
    "check_health",
    "inspect_list_models",
    # Errors
    "TSAgentKitError",
    "EContract",
    "ENoTSFM",
    "EInsufficient",
    "ETemporal",
]
