"""Core data structures for time-series forecasting.

Simplified from the original TSDataset + CovariateBundle + AlignedDataset
complexity into minimal, immutable data containers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.config import ForecastConfig


def _normalize_freq_alias(freq: str) -> str:
    """Normalize deprecated pandas aliases to current forms."""
    hourly_match = re.fullmatch(r"(\d*)H", freq)
    if hourly_match:
        return f"{hourly_match.group(1)}h"

    monthly_match = re.fullmatch(r"(\d*)M", freq)
    if monthly_match:
        return f"{monthly_match.group(1)}ME"

    return freq


@dataclass(frozen=True)
class CovariateSet:
    """Simplified covariate container.

    Replaces CovariateBundle and AlignedDataset with a minimal structure
    that just holds the three covariate types.
    """

    static: pd.DataFrame | None = None  # [unique_id, col1, col2, ...]
    past: pd.DataFrame | None = None  # [unique_id, ds, col1, ...]
    future: pd.DataFrame | None = None  # [unique_id, ds, col1, ...]

    def is_empty(self) -> bool:
        return self.static is None and self.past is None and self.future is None


@dataclass(frozen=True)
class TSDataset:
    """Minimal time-series dataset.

    Simplified from the original TSDataset which had complex hierarchy
    and covariate attachment mechanisms.
    """

    df: pd.DataFrame  # [unique_id, ds, y] + optional covariate columns
    config: ForecastConfig
    covariates: CovariateSet | None = None

    @property
    def n_series(self) -> int:
        return int(self.df["unique_id"].nunique())

    @property
    def min_length(self) -> int:
        return int(self.df.groupby("unique_id").size().min())

    @property
    def max_length(self) -> int:
        return int(self.df.groupby("unique_id").size().max())

    def get_series(self, unique_id: str) -> pd.DataFrame:
        """Extract single series by ID."""
        mask = self.df["unique_id"] == unique_id
        return self.df[mask].copy()

    def get_covariates_for_series(
        self,
        unique_id: str,
        covariate_type: Literal["static", "past", "future"],
    ) -> pd.DataFrame | None:
        """Extract covariates for a specific series.

        Args:
            unique_id: Series identifier
            covariate_type: Type of covariate to extract
                - "static": Static covariates (one row per series)
                - "past": Past-observed covariates (time-varying, historical only)
                - "future": Future-known covariates (time-varying, includes forecast horizon)

        Returns:
            DataFrame with covariates for the series, or None if not available.
            For static covariates: DataFrame with [unique_id, col1, col2, ...]
            For past/future covariates: DataFrame with [unique_id, ds, col1, col2, ...]
        """
        if self.covariates is None or self.covariates.is_empty():
            return None

        cov_df = None
        if covariate_type == "static":
            cov_df = self.covariates.static
        elif covariate_type == "past":
            cov_df = self.covariates.past
        elif covariate_type == "future":
            cov_df = self.covariates.future
        else:
            raise ValueError(f"Unknown covariate type: {covariate_type}")

        if cov_df is None:
            return None

        # Filter for the specific series
        if "unique_id" not in cov_df.columns:
            return None

        mask = cov_df["unique_id"] == unique_id
        result = cov_df[mask].copy()

        return result if len(result) > 0 else None

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        config: ForecastConfig,
        covariates: CovariateSet | None = None,
    ) -> TSDataset:
        """Create TSDataset from DataFrame with validation."""
        # Basic validation
        required = ["unique_id", "ds", "y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            from tsagentkit.core.errors import EContract

            raise EContract(
                f"Missing required columns: {missing}",
                context={"required": required, "found": list(df.columns)},
            )

        # Ensure sorted
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        return cls(df=df, config=config, covariates=covariates)

    def future_index(self, n_periods: int | None = None) -> pd.DataFrame:
        """Generate future timestamps for forecasting.

        Returns DataFrame with [unique_id, ds] for the forecast horizon.
        """
        h = n_periods or self.config.h
        ids = self.df["unique_id"].unique()

        # Get last timestamp per series
        last_ds = self.df.groupby("unique_id")["ds"].max()

        # Generate future dates
        future_rows = []
        freq = _normalize_freq_alias(self.config.freq)

        for uid in ids:
            last = last_ds[uid]
            future_dates = pd.date_range(start=last, periods=h + 1, freq=freq)[1:]
            for d in future_dates:
                future_rows.append({"unique_id": uid, "ds": d})

        return pd.DataFrame(future_rows)
