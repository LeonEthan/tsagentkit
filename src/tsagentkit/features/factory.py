"""Feature factory for point-in-time safe feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING
import warnings

import pandas as pd

from tsagentkit.features.covariates import CovariateManager
from tsagentkit.features.matrix import FeatureMatrix
from tsagentkit.features.versioning import FeatureConfig
from tsagentkit.features.extra.native import (
    build_native_feature_matrix,
    create_calendar_features,
    create_lag_features,
    create_observed_covariate_features,
    create_rolling_features,
)
from tsagentkit.features.tsfeatures_adapter import build_tsfeatures_matrix

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
        """Create features ensuring no lookahead bias."""
        engine = self._resolve_engine()
        config = self._resolved_config(engine)

        if engine == "tsfeatures":
            try:
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
        if self.config.engine == "auto":
            try:
                import tsfeatures  # type: ignore  # noqa: F401

                return "tsfeatures"
            except Exception:
                return "native"
        return self.config.engine

    def _resolved_config(self, engine: str) -> FeatureConfig:
        if engine == self.config.engine:
            return self.config
        return replace(self.config, engine=engine)

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        lags: list[int],
    ) -> pd.DataFrame:
        return create_lag_features(df, lags)

    def _create_calendar_features(
        self,
        df: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        return create_calendar_features(df, features)

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: dict[int, list[str]],
    ) -> pd.DataFrame:
        return create_rolling_features(df, windows)

    def _create_observed_covariate_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        return create_observed_covariate_features(df, self.config.observed_covariates)

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
