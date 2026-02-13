"""Covariate management for known vs observed covariates with leakage protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd

from tsagentkit.contracts.errors import ECovariateLeakage


class CovariatePolicy(Enum):
    """Policy for handling different covariate types.

    - KNOWN: Covariates known for all time steps (e.g., holidays, promotions planned in advance)
    - OBSERVED: Covariates only observed up to current time (e.g., actual sales, weather)
    """

    KNOWN = "known"
    OBSERVED = "observed"


@dataclass(frozen=True)
class CovariateConfig:
    """Configuration specifying covariate types.

    Attributes:
        known: List of column names for known covariates
        observed: List of column names for observed covariates

    Example:
        >>> config = CovariateConfig(
        ...     known=["holiday", "promotion_planned"],
        ...     observed=["competitor_price", "weather"],
        ... )
    """

    known: list[str] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate no overlap between known and observed."""
        overlap = set(self.known) & set(self.observed)
        if overlap:
            raise ValueError(f"Covariates cannot be both known and observed: {overlap}")

    def get_policy(self, column: str) -> CovariatePolicy | None:
        """Get the policy for a specific column.

        Args:
            column: Column name to check

        Returns:
            CovariatePolicy or None if column is not a covariate
        """
        if column in self.known:
            return CovariatePolicy.KNOWN
        elif column in self.observed:
            return CovariatePolicy.OBSERVED
        return None

    def all_covariates(self) -> list[str]:
        """Return all covariate column names."""
        return self.known + self.observed


def infer_covariate_config(
    df: pd.DataFrame,
    policy: str,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    target_col: str = "y",
) -> CovariateConfig:
    """Infer covariate configuration based on policy and data."""
    if policy == "ignore":
        return CovariateConfig()

    covariate_cols = [
        c for c in df.columns
        if c not in {id_col, ds_col, target_col}
    ]

    if not covariate_cols:
        return CovariateConfig()

    if policy == "known":
        return CovariateConfig(known=covariate_cols, observed=[])
    if policy == "observed":
        return CovariateConfig(known=[], observed=covariate_cols)

    # Auto policy: infer based on future rows (y is null)
    future_mask = df[target_col].isna() if target_col in df.columns else None
    known: list[str] = []
    observed: list[str] = []

    for col in covariate_cols:
        if future_mask is not None and future_mask.any():
            has_future_values = df.loc[future_mask, col].notna().any()
            if has_future_values:
                known.append(col)
            else:
                observed.append(col)
        else:
            # Default to observed if we can't see future values
            observed.append(col)

    return CovariateConfig(known=known, observed=observed)


class CovariateManager:
    """Manage known vs observed covariates with leakage protection.

    This class ensures that observed covariates are properly handled to prevent
    future information from leaking into training or predictions.

    Example:
        >>> manager = CovariateManager(
        ...     known_covariates=["holiday"],
        ...     observed_covariates=["promotion"],
        ... )
        >>>
        >>> # Validate no leakage
        >>> manager.validate_for_prediction(
        ...     df, forecast_start=datetime(2024, 1, 1), horizon=7
        ... )
        >>>
        >>> # Mask observed covariates for training
        >>> train_df = manager.mask_observed_for_training(df, target_col="y")
    """

    def __init__(
        self,
        known_covariates: list[str] | None = None,
        observed_covariates: list[str] | None = None,
    ):
        """Initialize the covariate manager.

        Args:
            known_covariates: Columns known for all time steps
            observed_covariates: Columns only observed up to current time
        """
        self.known_covariates = known_covariates or []
        self.observed_covariates = observed_covariates or []

        # Check for overlap
        overlap = set(self.known_covariates) & set(self.observed_covariates)
        if overlap:
            raise ValueError(f"Covariates cannot be both known and observed: {overlap}")

    def validate_for_prediction(
        self,
        df: pd.DataFrame,
        forecast_start: datetime,
        horizon: int,
        ds_col: str = "ds",
    ) -> None:
        """Validate that observed covariates don't leak future information.

        This checks that observed covariates do not have values beyond the
        forecast start time, which would indicate future information leakage.

        Args:
            df: DataFrame with covariates
            forecast_start: Start time of the forecast period
            horizon: Forecast horizon
            ds_col: Name of the timestamp column

        Raises:
            ECovariateLeakage: If observed covariates extend beyond forecast_start

        Example:
            >>> manager = CovariateManager(observed_covariates=["promo"])
            >>> # This will raise if promo has values after 2024-01-01
            >>> manager.validate_for_prediction(df, datetime(2024, 1, 1), horizon=7)
        """
        if not self.observed_covariates:
            return

        # Check each observed covariate
        for col in self.observed_covariates:
            if col not in df.columns:
                continue

            # Find rows where observed covariate has non-null values beyond forecast_start
            future_mask = (df[ds_col] >= forecast_start) & df[col].notna()
            future_count = future_mask.sum()

            if future_count > 0:
                raise ECovariateLeakage(
                    f"Observed covariate '{col}' has {future_count} values "
                    f"at or after forecast start time {forecast_start}. "
                    "Observed covariates cannot be known in advance.",
                    context={
                        "covariate": col,
                        "forecast_start": forecast_start.isoformat(),
                        "future_values_count": int(future_count),
                    },
                )

    def mask_observed_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
        ds_col: str = "ds",
        unique_id_col: str = "unique_id",
    ) -> pd.DataFrame:
        """Mask observed covariates at time t to prevent leakage during training.

        For observed covariates, we should only use values that would be available
        at prediction time. This means observed covariates at time t should be
        lagged (using values from before t) to prevent leakage.

        By default, this sets observed covariates to null for the target timestamp
        to ensure proper training. The caller is responsible for creating lagged
        versions of observed covariates before calling this method.

        Args:
            df: DataFrame with covariates
            target_col: Name of target column
            ds_col: Name of timestamp column
            unique_id_col: Name of unique_id column

        Returns:
            DataFrame with observed covariates masked at target time
        """
        if not self.observed_covariates:
            return df.copy()

        df = df.copy()

        # For training, we mask observed covariates at the prediction time
        # since they wouldn't be known yet. The model should use lagged versions.
        for col in self.observed_covariates:
            if col in df.columns:
                # Set to null - caller should create lagged features
                df[col] = None

        return df

    def create_lagged_observed_features(
        self,
        df: pd.DataFrame,
        lags: list[int],
        ds_col: str = "ds",
        unique_id_col: str = "unique_id",
    ) -> pd.DataFrame:
        """Create lagged versions of observed covariates.

        This creates lagged features for observed covariates to ensure
        point-in-time correctness. For a horizon h, observed covariates
        should be lagged by at least h to prevent leakage.

        Args:
            df: DataFrame with covariates
            lags: List of lag periods to create
            ds_col: Name of timestamp column
            unique_id_col: Name of unique_id column

        Returns:
            DataFrame with added lagged observed covariate columns
        """
        if not self.observed_covariates or not lags:
            return df.copy()

        df = df.copy()

        for col in self.observed_covariates:
            if col not in df.columns:
                continue

            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = (
                    df.groupby(unique_id_col)[col]
                    .shift(lag)
                    .values
                )

        return df

    def separate_covariates_for_prediction(
        self,
        df: pd.DataFrame,
        forecast_start: datetime,
        ds_col: str = "ds",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate known and observed covariates for prediction setup.

        Returns two DataFrames:
        1. Known covariates: Can be used directly (values known for all time steps)
        2. Observed covariates: Should be masked/handle carefully

        Args:
            df: DataFrame with covariates
            forecast_start: Start of forecast period
            ds_col: Name of timestamp column

        Returns:
            Tuple of (known_covariates_df, observed_covariates_df)
        """
        all_cols = ["unique_id", ds_col]

        known_cols = all_cols + [
            col for col in self.known_covariates if col in df.columns
        ]
        observed_cols = all_cols + [
            col for col in self.observed_covariates if col in df.columns
        ]

        known_df = df[known_cols].copy() if len(known_cols) > 2 else pd.DataFrame()
        observed_df = (
            df[observed_cols].copy() if len(observed_cols) > 2 else pd.DataFrame()
        )

        return known_df, observed_df

    def get_config(self) -> CovariateConfig:
        """Get covariate configuration."""
        return CovariateConfig(
            known=self.known_covariates,
            observed=self.observed_covariates,
        )
