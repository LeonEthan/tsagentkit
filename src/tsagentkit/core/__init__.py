"""Core module - unified contracts and data structures.

This module provides the foundational types and configurations for tsagentkit,
following a minimalist design philosophy.
"""

from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.data import CovariateSet, TSDataset
from tsagentkit.core.errors import (
    TSAgentKitError,
    EContractViolation,
    EDataQuality,
    EModelFailed,
    ETSFMRequired,
)
from tsagentkit.core.results import ForecastResult, RunResult

__all__ = [
    # Config
    "ForecastConfig",
    # Data
    "TSDataset",
    "CovariateSet",
    # Results
    "ForecastResult",
    "RunResult",
    # Errors
    "TSAgentKitError",
    "EContractViolation",
    "EDataQuality",
    "EModelFailed",
    "ETSFMRequired",
]
