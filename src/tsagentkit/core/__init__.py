"""Core module - unified contracts and data structures.

This module provides the foundational types and configurations for tsagentkit,
following a minimalist design philosophy.
"""

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
from tsagentkit.core.types import ModelArtifact

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
    "EContract",
    "ENoTSFM",
    "EInsufficient",
    "ETemporal",
    # Types
    "ModelArtifact",
]
