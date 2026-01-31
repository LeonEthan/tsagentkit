"""FeatureMatrix dataclass for storing engineered features with provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


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
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
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
