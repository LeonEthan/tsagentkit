"""tsagentkit - Minimalist time-series forecasting for AI agents.

Ultra-lightweight execution engine for time-series forecasting with TSFM ensemble.

Version 2.0.2 - Nanobot-Inspired Architecture

Input contract:
    DataFrame columns must be exactly: unique_id, ds, y.
    Custom column remapping is not supported.

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
    >>> from tsagentkit.models.registry import REGISTRY, list_models
    >>> df = validate(data)
    >>> dataset = TSDataset.from_dataframe(df, config)
    >>> models = [REGISTRY[m] for m in list_models(tsfm_only=True)]
    >>> ModelCache.preload(models)  # Load all TSFMs once
    >>> for batch in batches:
    ...     result = forecast(batch, h=7)  # Uses cached models
    >>> ModelCache.unload()  # Free memory when done

ModelCache unload semantics:
    ModelCache.unload() releases tsagentkit-owned model references, invokes
    adapter unload hooks, and performs best-effort backend cache cleanup.
    If user code still holds model references, Python cannot reclaim that memory.
"""

__version__ = "2.0.2"

# Core API
from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.dataset import CovariateSet, TSDataset
from tsagentkit.core.device import resolve_device
from tsagentkit.core.errors import (
    EContract,
    EInsufficient,
    ENoTSFM,
    ETemporal,
    TSAgentKitError,
)
from tsagentkit.core.results import ForecastResult, RunResult

# Inspection utilities
from tsagentkit.inspect import check_health

# Model Cache (for explicit lifecycle management)
from tsagentkit.models.cache import ModelCache
from tsagentkit.models.ensemble import ensemble_streaming
from tsagentkit.models.ensemble import ensemble_with_quantiles as ensemble
from tsagentkit.models.length_utils import (
    LengthAdjustment,
    adjust_context_length,
    check_data_compatibility,
    get_effective_limits,
    validate_prediction_length,
)

# Registry (for agent building)
from tsagentkit.models.registry import REGISTRY, ModelSpec, list_models

# Main entry points (Standard Pipeline)
# Agent Building (granular control)
from tsagentkit.pipeline import (
    build_dataset,
    fit_all,
    fit_all_parallel,
    forecast,
    make_plan,
    predict_all,
    predict_all_parallel,
    run_forecast,
    validate,
)

__all__ = [
    "__version__",
    # Standard Pipeline
    "forecast",
    "run_forecast",
    "ForecastConfig",
    "ForecastResult",
    "RunResult",
    "TSDataset",
    "CovariateSet",
    # Agent Building
    "validate",
    "build_dataset",
    "make_plan",
    "fit_all",
    "predict_all",
    "fit_all_parallel",
    "predict_all_parallel",
    "ensemble",
    "ensemble_streaming",
    # Model Cache
    "ModelCache",
    # Registry
    "REGISTRY",
    "ModelSpec",
    "list_models",
    # Length utilities
    "LengthAdjustment",
    "adjust_context_length",
    "validate_prediction_length",
    "get_effective_limits",
    "check_data_compatibility",
    # Device
    "resolve_device",
    # Inspection
    "check_health",
    # Errors
    "TSAgentKitError",
    "EContract",
    "ENoTSFM",
    "EInsufficient",
    "ETemporal",
]
