"""Feature engineering module for time series forecasting.

Provides point-in-time safe feature engineering with full versioning support.
"""

from __future__ import annotations

from tsagentkit.features.covariates import CovariateManager, CovariatePolicy
from tsagentkit.features.factory import FeatureConfig, FeatureFactory
from tsagentkit.features.matrix import FeatureMatrix
from tsagentkit.features.versioning import compute_feature_hash

__all__ = [
    "FeatureMatrix",
    "FeatureFactory",
    "FeatureConfig",
    "CovariateManager",
    "CovariatePolicy",
    "compute_feature_hash",
]
