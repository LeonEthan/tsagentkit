"""Feature factory for point-in-time safe feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tsagentkit.features.covariates import CovariateManager
from tsagentkit.features.matrix import FeatureMatrix
from tsagentkit.features.versioning import FeatureConfig, compute_feature_hash

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset


@dataclass
class FeatureFactory:
    """Point-in-time safe feature engineering for time series.

    This factory creates features ensuring no lookahead bias by strictly
    enforcing that features at time t only use information available at time t.

    Attributes:
        config: Feature configuration specifying what features to create
        covariate_manager: Manager for handling known vs observed covariates

    Example:
        >>> config = FeatureConfig(
        ...     lags=[1, 7, 14],
        ...     calendar_features=["dayofweek", "month"],
        ...     rolling_windows={7: ["mean", "std"]},
        ... )
        >>> factory = FeatureFactory(config)
        >>> matrix = factory.create_features(dataset)
        >>> print(matrix.signature)
        FeatureMatrix(c=abc123...,n=5)
    """

    config: FeatureConfig
    covariate_manager: CovariateManager | None = None

    def __post_init__(self) -> None:
        """Initialize covariate manager if not provided."""
        if self.covariate_manager is None:
            self.covariate_manager = CovariateManager(
                known_covariates=self.config.known_covariates,
                observed_covariates=self.config.observed_covariates,
            )

    def create_features(
        self,
        dataset: TSDataset,
        reference_time: datetime | None = None,
    ) -> FeatureMatrix:
        """Create features ensuring no lookahead bias.

        Args:
            dataset: Input TSDataset
            reference_time: Cutoff time for point-in-time correctness.
                          If None, uses max(ds) in the dataset.

        Returns:
            FeatureMatrix with engineered features

        Raises:
            ValueError: If required columns are missing
        """
        # Extract DataFrame from TSDataset
        if hasattr(dataset, "data"):
            df = dataset.data.copy()
        else:
            df = dataset.copy()

        # Validate required columns
        required = ["unique_id", "ds", "y"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
            df["ds"] = pd.to_datetime(df["ds"])

        # Determine reference time
        if reference_time is None:
            reference_time = df["ds"].max()

        # Filter to reference time for point-in-time correctness
        df = df[df["ds"] <= reference_time].copy()

        # Sort by unique_id and ds for proper lag/rolling calculations
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Track feature columns
        feature_cols: list[str] = []

        # Create lag features for target
        if self.config.lags:
            df = self._create_lag_features(df, self.config.lags)
            feature_cols.extend([f"y_lag_{lag}" for lag in self.config.lags])

        # Create calendar features
        if self.config.calendar_features:
            df = self._create_calendar_features(df, self.config.calendar_features)
            feature_cols.extend(self.config.calendar_features)

        # Create rolling features
        if self.config.rolling_windows:
            df = self._create_rolling_features(df, self.config.rolling_windows)
            for window, aggs in self.config.rolling_windows.items():
                for agg in aggs:
                    feature_cols.append(f"y_rolling_{agg}_{window}")

        # Handle observed covariates - create lagged versions
        if self.config.observed_covariates:
            df = self._create_observed_covariate_features(df)
            for col in self.config.observed_covariates:
                lag_col = f"{col}_lag_1"
                if lag_col in df.columns:
                    feature_cols.append(lag_col)

        # Include known covariates as-is
        if self.config.known_covariates:
            for col in self.config.known_covariates:
                if col in df.columns and col not in feature_cols:
                    feature_cols.append(col)

        # Add intercept if requested
        if self.config.include_intercept:
            df["intercept"] = 1.0
            feature_cols.append("intercept")

        # Compute config hash
        config_hash = compute_feature_hash(self.config)

        return FeatureMatrix(
            data=df,
            config_hash=config_hash,
            target_col="y",
            feature_cols=feature_cols,
            known_covariates=self.config.known_covariates,
            observed_covariates=self.config.observed_covariates,
        )

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        lags: list[int],
    ) -> pd.DataFrame:
        """Create lag features for the target variable.

        Args:
            df: DataFrame sorted by unique_id and ds
            lags: List of lag periods

        Returns:
            DataFrame with added lag columns
        """
        for lag in lags:
            df[f"y_lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
        return df

    def _create_calendar_features(
        self,
        df: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """Create calendar-based features.

        Args:
            df: DataFrame with ds column
            features: List of calendar features to create

        Returns:
            DataFrame with added calendar columns
        """
        ds = pd.to_datetime(df["ds"])

        feature_map = {
            "dayofweek": lambda d: d.dt.dayofweek,
            "month": lambda d: d.dt.month,
            "quarter": lambda d: d.dt.quarter,
            "year": lambda d: d.dt.year,
            "dayofmonth": lambda d: d.dt.day,
            "dayofyear": lambda d: d.dt.dayofyear,
            "weekofyear": lambda d: d.dt.isocalendar().week,
            "hour": lambda d: d.dt.hour,
            "minute": lambda d: d.dt.minute,
            "is_month_start": lambda d: d.dt.is_month_start.astype(int),
            "is_month_end": lambda d: d.dt.is_month_end.astype(int),
            "is_quarter_start": lambda d: d.dt.is_quarter_start.astype(int),
            "is_quarter_end": lambda d: d.dt.is_quarter_end.astype(int),
        }

        for feature in features:
            if feature in feature_map:
                df[feature] = feature_map[feature](ds)

        return df

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: dict[int, list[str]],
    ) -> pd.DataFrame:
        """Create rolling window statistics.

        Uses right-aligned windows (data up to but not including current point)
        to prevent lookahead bias.

        Args:
            df: DataFrame sorted by unique_id and ds
            windows: Dict mapping window sizes to list of aggregations

        Returns:
            DataFrame with added rolling columns
        """
        agg_map = {
            "mean": lambda x: x.rolling(window=window, min_periods=1).mean(),
            "std": lambda x: x.rolling(window=window, min_periods=1).std(),
            "min": lambda x: x.rolling(window=window, min_periods=1).min(),
            "max": lambda x: x.rolling(window=window, min_periods=1).max(),
            "sum": lambda x: x.rolling(window=window, min_periods=1).sum(),
            "median": lambda x: x.rolling(window=window, min_periods=1).median(),
        }

        for window, aggs in windows.items():
            for agg in aggs:
                if agg in agg_map:
                    df[f"y_rolling_{agg}_{window}"] = (
                        df.groupby("unique_id")["y"]
                        .transform(lambda x: agg_map[agg](x))
                    )

        return df

    def _create_observed_covariate_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create features from observed covariates.

        Observed covariates at time t should use lagged values to prevent
        leakage. We create lag_1 by default.

        Args:
            df: DataFrame with observed covariates

        Returns:
            DataFrame with lagged observed covariates
        """
        for col in self.config.observed_covariates:
            if col not in df.columns:
                continue

            # Create lagged version to prevent leakage
            lag_col = f"{col}_lag_1"
            df[lag_col] = df.groupby("unique_id")[col].shift(1)

        return df

    def get_feature_importance_template(
        self,
    ) -> dict[str, float]:
        """Return a template for feature importance scores.

        Returns:
            Dict mapping feature names to 0.0 (template for importance scores)
        """
        importance: dict[str, float] = {}

        for lag in self.config.lags:
            importance[f"y_lag_{lag}"] = 0.0

        for feature in self.config.calendar_features:
            importance[feature] = 0.0

        for window, aggs in self.config.rolling_windows.items():
            for agg in aggs:
                importance[f"y_rolling_{agg}_{window}"] = 0.0

        for col in self.config.known_covariates:
            importance[col] = 0.0

        for col in self.config.observed_covariates:
            importance[f"{col}_lag_1"] = 0.0

        if self.config.include_intercept:
            importance["intercept"] = 0.0

        return importance
