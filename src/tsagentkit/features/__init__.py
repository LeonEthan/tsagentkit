"""Feature engineering module for time series forecasting.

Provides point-in-time safe feature engineering with full versioning support.
"""

from __future__ import annotations

from tsagentkit.features.config import (
    FeatureConfig,
    FeatureMatrix,
    compute_feature_hash,
    config_to_dict,
    config_from_dict,
    configs_equal,
)
from tsagentkit.features.covariates import CovariateManager, CovariatePolicy
from tsagentkit.features.engine import FeatureFactory, build_native_feature_matrix

__all__ = [
    "FeatureMatrix",
    "FeatureFactory",
    "FeatureConfig",
    "CovariateManager",
    "CovariatePolicy",
    "compute_feature_hash",
    "config_to_dict",
    "config_from_dict",
    "configs_equal",
    "build_native_feature_matrix",
]
