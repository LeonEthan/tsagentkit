"""TSDataset implementation.

Immutable wrapper around DataFrame for time series data with
guaranteed schema and temporal integrity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.contracts import TaskSpec, validate_contract
from tsagentkit.contracts.results import ValidationReport

from .sparsity import SparsityProfile, compute_sparsity_profile

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class TSDataset:
    """Immutable time series dataset container.

    Wraps a DataFrame with guaranteed schema and provides methods
    for time series operations. All operations return new instances.

    Attributes:
        df: The underlying DataFrame (guaranteed sorted by unique_id, ds)
        task_spec: Task specification for this dataset
        sparsity_profile: Computed sparsity profile
        metadata: Additional dataset metadata

    Examples:
        >>> from tsagentkit import TaskSpec
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "unique_id": ["A", "A", "B", "B"],
        ...     "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
        ...     "y": [1.0, 2.0, 3.0, 4.0],
        ... })
        >>> spec = TaskSpec(horizon=7, freq="D")
        >>> dataset = TSDataset.from_dataframe(df, spec)
    """

    df: pd.DataFrame
    task_spec: TaskSpec
    sparsity_profile: SparsityProfile | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the dataset after creation."""
        # Since dataclass is frozen, we can't modify, but we can validate
        required_cols = {"unique_id", "ds", "y"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"TSDataset missing required columns: {missing}")

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df["ds"]):
            raise ValueError("Column 'ds' must be datetime type")

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        task_spec: TaskSpec,
        validate: bool = True,
        compute_sparsity: bool = True,
    ) -> "TSDataset":
        """Create TSDataset from DataFrame.

        Args:
            data: Input DataFrame
            task_spec: Task specification
            validate: Whether to validate input (default: True)
            compute_sparsity: Whether to compute sparsity profile (default: True)

        Returns:
            New TSDataset instance

        Raises:
            ValueError: If validation fails
        """
        df = data.copy()

        # Validate if requested
        if validate:
            report = validate_contract(df)
            if not report.valid:
                report.raise_if_errors()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
            df["ds"] = pd.to_datetime(df["ds"])

        # Sort by unique_id, ds
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Compute sparsity profile
        sparsity = None
        if compute_sparsity:
            sparsity = compute_sparsity_profile(df)

        return cls(
            df=df,
            task_spec=task_spec,
            sparsity_profile=sparsity,
        )

    @property
    def n_series(self) -> int:
        """Number of unique series."""
        return self.df["unique_id"].nunique()

    @property
    def n_observations(self) -> int:
        """Total number of observations."""
        return len(self.df)

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Date range of the dataset."""
        return (self.df["ds"].min(), self.df["ds"].max())

    @property
    def series_ids(self) -> list[str]:
        """List of unique series IDs."""
        return sorted(self.df["unique_id"].unique().tolist())

    @property
    def freq(self) -> str:
        """Frequency from task spec."""
        return self.task_spec.freq

    def get_series(self, unique_id: str) -> pd.DataFrame:
        """Get data for a single series.

        Args:
            unique_id: Series identifier

        Returns:
            DataFrame with series data
        """
        return self.df[self.df["unique_id"] == unique_id].copy()

    def filter_series(self, series_ids: list[str]) -> "TSDataset":
        """Create new dataset with only specified series.

        Args:
            series_ids: List of series IDs to keep

        Returns:
            New TSDataset instance
        """
        mask = self.df["unique_id"].isin(series_ids)
        new_df = self.df[mask].copy()

        return TSDataset(
            df=new_df,
            task_spec=self.task_spec,
            sparsity_profile=self.sparsity_profile,  # Keep original profile
            metadata=self.metadata.copy(),
        )

    def filter_dates(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "TSDataset":
        """Create new dataset filtered by date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            New TSDataset instance
        """
        mask = pd.Series(True, index=self.df.index)

        if start is not None:
            start_ts = pd.to_datetime(start)
            mask &= self.df["ds"] >= start_ts

        if end is not None:
            end_ts = pd.to_datetime(end)
            mask &= self.df["ds"] <= end_ts

        new_df = self.df[mask].copy()

        return TSDataset(
            df=new_df,
            task_spec=self.task_spec,
            sparsity_profile=None,  # Need to recompute
            metadata=self.metadata.copy(),
        )

    def split_train_test(
        self,
        test_size: int | None = None,
        test_start: str | pd.Timestamp | None = None,
    ) -> tuple["TSDataset", "TSDataset"]:
        """Split dataset into train and test sets.

        Temporal split - uses cutoff date or last N observations per series.

        Args:
            test_size: Number of observations for test (per series)
            test_start: Start date for test set

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            ValueError: If neither test_size nor test_start provided
        """
        if test_start is not None:
            # Use date-based split
            cutoff = pd.to_datetime(test_start)
            train_mask = self.df["ds"] < cutoff
        elif test_size is not None:
            # Use last N observations per series
            train_mask = pd.Series(False, index=self.df.index)

            for uid in self.df["unique_id"].unique():
                series_idx = self.df[self.df["unique_id"] == uid].index
                if len(series_idx) > test_size:
                    train_idx = series_idx[:-test_size]
                    train_mask.loc[train_idx] = True
        else:
            raise ValueError("Must provide either test_size or test_start")

        train_df = self.df[train_mask].copy()
        test_df = self.df[~train_mask].copy()

        train_ds = TSDataset(
            df=train_df,
            task_spec=self.task_spec,
            sparsity_profile=None,  # Need to recompute
            metadata=self.metadata.copy(),
        )

        test_ds = TSDataset(
            df=test_df,
            task_spec=self.task_spec,
            sparsity_profile=None,
            metadata=self.metadata.copy(),
        )

        return train_ds, test_ds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Note: DataFrame is converted to records format.
        """
        return {
            "df": self.df.to_dict("records"),
            "task_spec": self.task_spec.model_dump(),
            "sparsity_profile": self.sparsity_profile.series_profiles if self.sparsity_profile else None,
            "metadata": self.metadata,
        }


def build_dataset(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    validate: bool = True,
    compute_sparsity: bool = True,
) -> TSDataset:
    """Build a TSDataset from raw data.

    Convenience function for creating TSDataset.

    Args:
        data: Input DataFrame
        task_spec: Task specification
        validate: Whether to validate input (default: True)
        compute_sparsity: Whether to compute sparsity profile (default: True)

    Returns:
        New TSDataset instance
    """
    return TSDataset.from_dataframe(
        data=data,
        task_spec=task_spec,
        validate=validate,
        compute_sparsity=compute_sparsity,
    )
