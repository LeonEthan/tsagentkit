"""Feature engineering engine with native backend.

This module consolidates the FeatureFactory and native feature engineering
backend into a single cohesive module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.features.config import (
    FeatureConfig,
    FeatureMatrix,
    compute_feature_hash,
)
from tsagentkit.features.covariates import CovariateManager

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset


def extract_panel(dataset: Any) -> pd.DataFrame:
    """Extract DataFrame from dataset object."""
    if hasattr(dataset, "df"):
        return dataset.df.copy()
    if hasattr(dataset, "data"):
        return dataset.data.copy()
    return dataset.copy()


def prepare_panel(df: pd.DataFrame, reference_time: datetime | None) -> pd.DataFrame:
    """Prepare panel data for feature engineering.

    Validates required columns and filters to reference_time.
    """
    required = ["unique_id", "ds", "y"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    if reference_time is None:
        reference_time = df["ds"].max()

    df = df[df["ds"] <= reference_time].copy()
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def create_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Create lag features for target variable."""
    for lag in lags:
        df[f"y_lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
    return df


def create_calendar_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Create calendar-based features from datetime index."""
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


def create_rolling_features(df: pd.DataFrame, windows: dict[int, list[str]]) -> pd.DataFrame:
    """Create rolling window features for target variable."""
    for window, aggs in windows.items():
        for agg in aggs:
            if agg == "mean":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            elif agg == "std":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .std()
                )
            elif agg == "min":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .min()
                )
            elif agg == "max":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .max()
                )
            elif agg == "sum":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .sum()
                )
            elif agg == "median":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .median()
                )
            else:
                continue
            df[f"y_rolling_{agg}_{window}"] = series

    return df


def create_observed_covariate_features(
    df: pd.DataFrame,
    observed_covariates: list[str],
) -> pd.DataFrame:
    """Create lagged features for observed covariates."""
    for col in observed_covariates:
        if col not in df.columns:
            continue
        lag_col = f"{col}_lag_1"
        df[lag_col] = df.groupby("unique_id")[col].shift(1)
    return df


def build_native_feature_matrix(
    dataset: Any,
    config: FeatureConfig,
    reference_time: datetime | None = None,
) -> FeatureMatrix:
    """Build feature matrix using native feature engineering backend.

    Args:
        dataset: Dataset object or DataFrame with time series data
        config: Feature configuration
        reference_time: Optional reference time for point-in-time features

    Returns:
        FeatureMatrix with engineered features
    """
    df = prepare_panel(extract_panel(dataset), reference_time)

    feature_cols: list[str] = []

    if config.lags:
        df = create_lag_features(df, config.lags)
        feature_cols.extend([f"y_lag_{lag}" for lag in config.lags])

    if config.calendar_features:
        df = create_calendar_features(df, config.calendar_features)
        feature_cols.extend(config.calendar_features)

    if config.rolling_windows:
        df = create_rolling_features(df, config.rolling_windows)
        for window, aggs in config.rolling_windows.items():
            for agg in aggs:
                feature_cols.append(f"y_rolling_{agg}_{window}")

    if config.observed_covariates:
        df = create_observed_covariate_features(df, config.observed_covariates)
        for col in config.observed_covariates:
            lag_col = f"{col}_lag_1"
            if lag_col in df.columns:
                feature_cols.append(lag_col)

    if config.known_covariates:
        for col in config.known_covariates:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)

    if config.include_intercept:
        df["intercept"] = 1.0
        feature_cols.append("intercept")

    config_hash = compute_feature_hash(config)

    return FeatureMatrix(
        data=df,
        config_hash=config_hash,
        target_col="y",
        feature_cols=feature_cols,
        known_covariates=config.known_covariates,
        observed_covariates=config.observed_covariates,
    )


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
        """Create features ensuring no lookahead bias."""
        engine = self._resolve_engine()
        config = self._resolved_config(engine)

        if engine == "tsfeatures":
            try:
                from tsagentkit.features.tsfeatures import build_tsfeatures_matrix

                return build_tsfeatures_matrix(
                    dataset=dataset,
                    config=config,
                    reference_time=reference_time,
                )
            except ImportError as exc:
                if not config.allow_fallback:
                    raise
                warnings.warn(
                    f"tsfeatures unavailable ({exc}); falling back to native features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                config = self._resolved_config("native")
                return build_native_feature_matrix(
                    dataset=dataset,
                    config=config,
                    reference_time=reference_time,
                )

        return build_native_feature_matrix(
            dataset=dataset,
            config=config,
            reference_time=reference_time,
        )

    def _resolve_engine(self) -> str:
        """Resolve feature engine based on configuration.

        When engine is "auto", defaults to tsfeatures. If tsfeatures is not
        available, raises ImportError unless allow_fallback is True.
        """
        if self.config.engine == "auto":
            try:
                import tsfeatures  # type: ignore  # noqa: F401

                return "tsfeatures"
            except Exception as exc:
                if not self.config.allow_fallback:
                    raise ImportError(
                        "tsfeatures is required but not installed. "
                        "Install with: pip install tsfeatures "
                        "Or set allow_fallback=True to use native features."
                    ) from exc
                warnings.warn(
                    "tsfeatures is not installed; falling back to native features. "
                    "For reproducibility, install tsfeatures or explicitly set engine='native'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return "native"
        return self.config.engine

    def _resolved_config(self, engine: str) -> FeatureConfig:
        if engine == self.config.engine:
            return self.config
        return replace(self.config, engine=engine)

    # Backward compatibility: delegate to standalone functions
    def _create_lag_features(
        self,
        df: pd.DataFrame,
        lags: list[int],
    ) -> pd.DataFrame:
        """Create lag features."""
        return create_lag_features(df, lags)

    def _create_calendar_features(
        self,
        df: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """Create calendar features."""
        return create_calendar_features(df, features)

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: dict[int, list[str]],
    ) -> pd.DataFrame:
        """Create rolling window features."""
        return create_rolling_features(df, windows)

    def _create_observed_covariate_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create observed covariate lag features."""
        return create_observed_covariate_features(df, self.config.observed_covariates)

    def get_feature_importance_template(self) -> dict[str, float]:
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
