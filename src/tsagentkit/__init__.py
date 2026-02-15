"""tsagentkit - Minimalist time-series forecasting for AI agents.

This library provides a strict, production-grade execution engine for
time-series forecasting with TSFM-first strategy and automatic fallback.

Version 2.0.0 - Minimalist Refactoring

Basic usage:
    >>> from tsagentkit import forecast
    >>> result = forecast(data, h=7)
    >>> print(result.forecast.df)

Advanced usage:
    >>> from tsagentkit import ForecastConfig, run_pipeline
    >>> config = ForecastConfig.quick(h=14, freq='H')
    >>> result = run_pipeline(data, config)
"""

__version__ = "2.0.0"

# Core API
from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.data import CovariateSet, TSDataset
from tsagentkit.core.results import ForecastResult, RunResult
from tsagentkit.core.errors import (
    TSAgentKitError,
    EContractViolation,
    EDataQuality,
    EModelFailed,
    ETSFMRequired,
)

# Main entry point
from tsagentkit.pipeline.runner import forecast, run_pipeline

# Pipeline stages
from tsagentkit.pipeline.stages import STAGES, PipelineStage

# Router
from tsagentkit.router import (
    ModelCandidate,
    Plan,
    build_plan,
    inspect_tsfm_adapters,
)

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
    "Plan",
    "ModelCandidate",
    "build_plan",
    "inspect_tsfm_adapters",
    "STAGES",
    "PipelineStage",
    # Errors
    "TSAgentKitError",
    "EContractViolation",
    "EDataQuality",
    "EModelFailed",
    "ETSFMRequired",
]
