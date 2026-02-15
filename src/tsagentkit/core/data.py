"""Core data structures for time-series forecasting.

Simplified from the original TSDataset + CovariateBundle + AlignedDataset
complexity into minimal, immutable data containers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.config import ForecastConfig


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
        return self.df[self.config.id_col].nunique()

    @property
    def min_length(self) -> int:
        return self.df.groupby(self.config.id_col).size().min()

    @property
    def max_length(self) -> int:
        return self.df.groupby(self.config.id_col).size().max()

    def get_series(self, unique_id: str) -> pd.DataFrame:
        """Extract single series by ID."""
        mask = self.df[self.config.id_col] == unique_id
        return self.df[mask].copy()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        config: ForecastConfig,
        covariates: CovariateSet | None = None,
    ) -> TSDataset:
        """Create TSDataset from DataFrame with validation."""
        # Basic validation
        required = [config.id_col, config.time_col, config.target_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            from tsagentkit.core.errors import EContractViolation

            raise EContractViolation(
                f"Missing required columns: {missing}",
                context={"required": required, "found": list(df.columns)},
            )

        # Ensure sorted
        df = df.sort_values([config.id_col, config.time_col]).reset_index(drop=True)

        return cls(df=df, config=config, covariates=covariates)

    def future_index(self, n_periods: int | None = None) -> pd.DataFrame:
        """Generate future timestamps for forecasting.

        Returns DataFrame with [unique_id, ds] for the forecast horizon.
        """
        h = n_periods or self.config.h
        ids = self.df[self.config.id_col].unique()

        # Get last timestamp per series
        last_ds = self.df.groupby(self.config.id_col)[self.config.time_col].max()

        # Generate future dates
        future_rows = []
        freq = self.config.freq

        for uid in ids:
            last = last_ds[uid]
            future_dates = pd.date_range(start=last, periods=h + 1, freq=freq)[1:]
            for d in future_dates:
                future_rows.append({self.config.id_col: uid, self.config.time_col: d})

        return pd.DataFrame(future_rows)
