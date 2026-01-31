"""Feature versioning and hashing for provenance tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering.

    Attributes:
        lags: List of lag periods to create (e.g., [1, 7, 14])
        calendar_features: List of calendar features to create
                         (e.g., ["dayofweek", "month", "quarter", "year"])
        rolling_windows: Dict mapping window sizes to aggregation functions
                        (e.g., {7: ["mean", "std"], 30: ["mean"]})
        known_covariates: List of column names for known covariates
        observed_covariates: List of column names for observed covariates
        include_intercept: Whether to include an intercept column (all 1s)

    Example:
        >>> config = FeatureConfig(
        ...     lags=[1, 7, 14],
        ...     calendar_features=["dayofweek", "month"],
        ...     rolling_windows={7: ["mean", "std"], 30: ["mean"]},
        ...     known_covariates=["holiday"],
        ...     observed_covariates=["promotion"],
        ... )
        >>> print(compute_feature_hash(config))
        abc123def456...
    """

    lags: list[int] = field(default_factory=list)
    calendar_features: list[str] = field(default_factory=list)
    rolling_windows: dict[int, list[str]] = field(default_factory=dict)
    known_covariates: list[str] = field(default_factory=list)
    observed_covariates: list[str] = field(default_factory=list)
    include_intercept: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after creation."""
        # Validate lags are positive integers
        for lag in self.lags:
            if not isinstance(lag, int) or lag < 1:
                raise ValueError(f"Lags must be positive integers, got {lag}")

        # Validate calendar features
        valid_calendar = {
            "dayofweek", "month", "quarter", "year", "dayofmonth",
            "dayofyear", "weekofyear", "hour", "minute", "is_month_start",
            "is_month_end", "is_quarter_start", "is_quarter_end",
        }
        invalid = set(self.calendar_features) - valid_calendar
        if invalid:
            raise ValueError(f"Invalid calendar features: {invalid}. Valid: {valid_calendar}")

        # Validate rolling aggregations
        valid_aggs = {"mean", "std", "min", "max", "sum", "median"}
        for window, aggs in self.rolling_windows.items():
            if not isinstance(window, int) or window < 1:
                raise ValueError(f"Window sizes must be positive integers, got {window}")
            invalid_aggs = set(aggs) - valid_aggs
            if invalid_aggs:
                raise ValueError(f"Invalid aggregations: {invalid_aggs}. Valid: {valid_aggs}")

        # Check for overlap in covariates
        overlap = set(self.known_covariates) & set(self.observed_covariates)
        if overlap:
            raise ValueError(f"Covariates cannot be both known and observed: {overlap}")


def compute_feature_hash(config: FeatureConfig) -> str:
    """Compute deterministic hash of feature configuration.

    The hash includes all feature configuration parameters to ensure
    that any change to the feature engineering setup results in a
    different hash for provenance tracking.

    Args:
        config: Feature configuration to hash

    Returns:
        SHA-256 hash string (truncated to 16 characters)

    Example:
        >>> config = FeatureConfig(lags=[1, 7], calendar_features=["dayofweek"])
        >>> h = compute_feature_hash(config)
        >>> len(h)
        16
    """
    # Build normalized configuration dict
    config_dict = {
        "lags": sorted(config.lags) if config.lags else [],
        "calendar": sorted(config.calendar_features),
        "rolling": [
            {"window": w, "aggs": sorted(a)}
            for w, a in sorted(config.rolling_windows.items())
        ],
        "known_covariates": sorted(config.known_covariates),
        "observed_covariates": sorted(config.observed_covariates),
        "include_intercept": config.include_intercept,
    }

    # Create deterministic JSON representation
    json_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))

    # Compute hash
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def configs_equal(config1: FeatureConfig, config2: FeatureConfig) -> bool:
    """Check if two feature configurations are equivalent.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        True if configurations produce identical features
    """
    return compute_feature_hash(config1) == compute_feature_hash(config2)


def config_to_dict(config: FeatureConfig) -> dict[str, Any]:
    """Convert feature config to dictionary for serialization.

    Args:
        config: Feature configuration

    Returns:
        Dictionary representation
    """
    return {
        "lags": config.lags,
        "calendar_features": config.calendar_features,
        "rolling_windows": config.rolling_windows,
        "known_covariates": config.known_covariates,
        "observed_covariates": config.observed_covariates,
        "include_intercept": config.include_intercept,
    }


def config_from_dict(data: dict[str, Any]) -> FeatureConfig:
    """Create feature config from dictionary.

    Args:
        data: Dictionary with configuration values

    Returns:
        FeatureConfig instance
    """
    return FeatureConfig(
        lags=data.get("lags", []),
        calendar_features=data.get("calendar_features", []),
        rolling_windows=data.get("rolling_windows", {}),
        known_covariates=data.get("known_covariates", []),
        observed_covariates=data.get("observed_covariates", []),
        include_intercept=data.get("include_intercept", False),
    )
