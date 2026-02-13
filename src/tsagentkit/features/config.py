"""Feature configuration and matrix for engineered features.

This module consolidates feature configuration (FeatureConfig) and the feature
matrix container (FeatureMatrix) along with versioning/hashing utilities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering.

    Attributes:
        engine: Feature backend selection ("auto", "native", "tsfeatures")
        lags: List of lag periods to create (e.g., [1, 7, 14])
        calendar_features: List of calendar features to create
                         (e.g., ["dayofweek", "month", "quarter", "year"])
        rolling_windows: Dict mapping window sizes to aggregation functions
                        (e.g., {7: ["mean", "std"], 30: ["mean"]})
        known_covariates: List of column names for known covariates
        observed_covariates: List of column names for observed covariates
        include_intercept: Whether to include an intercept column (all 1s)
        tsfeatures_features: Optional list of tsfeatures function names
        tsfeatures_freq: Optional season length to pass to tsfeatures
        tsfeatures_dict_freqs: Optional dict mapping pandas freq -> season length
        allow_fallback: Allow fallback to native when tsfeatures is unavailable

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

    engine: Literal["auto", "native", "tsfeatures"] = "auto"
    lags: list[int] = field(default_factory=list)
    calendar_features: list[str] = field(default_factory=list)
    rolling_windows: dict[int, list[str]] = field(default_factory=dict)
    known_covariates: list[str] = field(default_factory=list)
    observed_covariates: list[str] = field(default_factory=list)
    include_intercept: bool = False
    tsfeatures_features: list[str] = field(default_factory=list)
    tsfeatures_freq: int | None = None
    tsfeatures_dict_freqs: dict[str, int] = field(default_factory=dict)
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after creation."""
        if self.engine not in {"auto", "native", "tsfeatures"}:
            raise ValueError(f"Invalid feature engine: {self.engine}")

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

        if self.tsfeatures_freq is not None and self.tsfeatures_freq < 1:
            raise ValueError("tsfeatures_freq must be a positive integer when provided")

        for key, value in self.tsfeatures_dict_freqs.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"tsfeatures_dict_freqs must map to positive integers, got {key}={value}"
                )


@dataclass(frozen=True)
class FeatureMatrix:
    """Container for engineered features with provenance.

    Attributes:
        data: DataFrame with engineered features (includes unique_id, ds, target)
        config_hash: Hash of the feature configuration used to create these features
        target_col: Name of the target variable column
        feature_cols: List of engineered feature column names
        known_covariates: List of known covariate column names
        observed_covariates: List of observed covariate column names
        created_at: ISO 8601 timestamp of feature matrix creation

    Example:
        >>> matrix = FeatureMatrix(
        ...     data=df_with_features,
        ...     config_hash="abc123...",
        ...     feature_cols=["lag_7", "rolling_mean_30", "dayofweek"],
        ...     known_covariates=["holiday"],
        ...     observed_covariates=["promotion"],
        ... )
        >>> print(matrix.signature)
        FeatureMatrix(c=abc123...,n=3)
    """

    data: pd.DataFrame
    config_hash: str
    target_col: str = "y"
    feature_cols: list[str] = field(default_factory=list)
    known_covariates: list[str] = field(default_factory=list)
    observed_covariates: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate the feature matrix after creation."""
        # Validate required columns exist
        required = ["unique_id", "ds", self.target_col]
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Validate feature columns exist in data
        invalid_features = [col for col in self.feature_cols if col not in self.data.columns]
        if invalid_features:
            raise ValueError(f"Feature columns not in data: {invalid_features}")

        # Validate covariate columns exist
        invalid_known = [col for col in self.known_covariates if col not in self.data.columns]
        if invalid_known:
            raise ValueError(f"Known covariates not in data: {invalid_known}")

        invalid_observed = [
            col for col in self.observed_covariates if col not in self.data.columns
        ]
        if invalid_observed:
            raise ValueError(f"Observed covariates not in data: {invalid_observed}")

    @property
    def signature(self) -> str:
        """Return feature matrix signature for provenance.

        Returns:
            String signature like "FeatureMatrix(c=abc123...,n=5)"
        """
        return f"FeatureMatrix(c={self.config_hash},n={len(self.feature_cols)})"

    def to_pandas(self) -> pd.DataFrame:
        """Return the feature matrix as a pandas DataFrame.

        Returns:
            Copy of the underlying DataFrame
        """
        return self.data.copy()

    def get_feature_data(self) -> pd.DataFrame:
        """Get only the feature columns (excluding id, timestamp, target).

        Returns:
            DataFrame with only feature columns
        """
        return self.data[self.feature_cols].copy()

    def get_target_data(self) -> pd.Series:
        """Get the target variable.

        Returns:
            Series with target values
        """
        return self.data[self.target_col].copy()

    def get_covariate_data(self, covariate_type: str | None = None) -> pd.DataFrame:
        """Get covariate columns.

        Args:
            covariate_type: "known", "observed", or None (all covariates)

        Returns:
            DataFrame with covariate columns
        """
        if covariate_type == "known":
            cols = self.known_covariates
        elif covariate_type == "observed":
            cols = self.observed_covariates
        else:
            cols = self.known_covariates + self.observed_covariates

        if not cols:
            return pd.DataFrame(index=self.data.index)

        return self.data[cols].copy()

    def validate(self) -> list[str]:
        """Validate the feature matrix and return any issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check for nulls in features
        if self.feature_cols:
            null_counts = self.data[self.feature_cols].isnull().sum()
            if null_counts.any():
                cols_with_nulls = null_counts[null_counts > 0].index.tolist()
                issues.append(f"Features contain nulls: {cols_with_nulls}")

        # Check for infinite values
        if self.feature_cols:
            numeric_cols = self.data[self.feature_cols].select_dtypes(include=["number"])
            if numeric_cols is not None and not numeric_cols.empty:
                inf_counts = np.isinf(numeric_cols).sum()
                if inf_counts.any():
                    cols_with_inf = inf_counts[inf_counts > 0].index.tolist()
                    issues.append(f"Features contain infinite values: {cols_with_inf}")

        # Check target exists and is numeric
        if self.target_col not in self.data.columns:
            issues.append(f"Target column '{self.target_col}' not found")
        elif not pd.api.types.is_numeric_dtype(self.data[self.target_col]):
            issues.append(f"Target column '{self.target_col}' is not numeric")

        return issues


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
        "engine": config.engine,
        "lags": sorted(config.lags) if config.lags else [],
        "calendar": sorted(config.calendar_features),
        "rolling": [
            {"window": w, "aggs": sorted(a)}
            for w, a in sorted(config.rolling_windows.items())
        ],
        "known_covariates": sorted(config.known_covariates),
        "observed_covariates": sorted(config.observed_covariates),
        "include_intercept": config.include_intercept,
        "tsfeatures_features": sorted(config.tsfeatures_features),
        "tsfeatures_freq": config.tsfeatures_freq,
        "tsfeatures_dict_freqs": {
            k: config.tsfeatures_dict_freqs[k]
            for k in sorted(config.tsfeatures_dict_freqs)
        },
        "allow_fallback": config.allow_fallback,
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


def config_to_dict(config: FeatureConfig) -> dict[str, object]:
    """Convert feature config to dictionary for serialization.

    Args:
        config: Feature configuration

    Returns:
        Dictionary representation
    """
    return {
        "engine": config.engine,
        "lags": config.lags,
        "calendar_features": config.calendar_features,
        "rolling_windows": config.rolling_windows,
        "known_covariates": config.known_covariates,
        "observed_covariates": config.observed_covariates,
        "include_intercept": config.include_intercept,
        "tsfeatures_features": config.tsfeatures_features,
        "tsfeatures_freq": config.tsfeatures_freq,
        "tsfeatures_dict_freqs": config.tsfeatures_dict_freqs,
        "allow_fallback": config.allow_fallback,
    }


def config_from_dict(data: dict[str, object]) -> FeatureConfig:
    """Create feature config from dictionary.

    Args:
        data: Dictionary with configuration values

    Returns:
        FeatureConfig instance
    """
    return FeatureConfig(
        engine=cast(
            Literal["auto", "native", "tsfeatures"],
            data.get("engine", "auto"),
        ),
        lags=cast(list[int], data.get("lags", [])),
        calendar_features=cast(list[str], data.get("calendar_features", [])),
        rolling_windows=cast(dict[int, list[str]], data.get("rolling_windows", {})),
        known_covariates=cast(list[str], data.get("known_covariates", [])),
        observed_covariates=cast(list[str], data.get("observed_covariates", [])),
        include_intercept=cast(bool, data.get("include_intercept", False)),
        tsfeatures_features=cast(list[str], data.get("tsfeatures_features", [])),
        tsfeatures_freq=cast(int | None, data.get("tsfeatures_freq")),
        tsfeatures_dict_freqs=cast(dict[str, int], data.get("tsfeatures_dict_freqs", {})),
        allow_fallback=cast(bool, data.get("allow_fallback", True)),
    )
