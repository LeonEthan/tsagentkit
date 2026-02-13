"""Temporal split and cutoff utilities for backtesting."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from tsagentkit.contracts import ESplitRandomForbidden


def validate_temporal_ordering(df: pd.DataFrame) -> None:
    """Validate that data is temporally ordered (no shuffling)."""
    expected_order = df.sort_values(["unique_id", "ds"]).index
    if not df.index.equals(expected_order):
        raise ESplitRandomForbidden(
            "Data must be sorted by (unique_id, ds). "
            "Random splits or shuffling is strictly forbidden.",
            context={
                "suggestion": "Ensure data is sorted: df.sort_values(['unique_id', 'ds'])",
            },
        )

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]
        dates = pd.to_datetime(series["ds"])
        if not dates.is_monotonic_increasing:
            raise ESplitRandomForbidden(
                f"Dates for series {uid} are not monotonically increasing. "
                f"Data may be shuffled or contain time-travel.",
                context={"series_id": uid},
            )


def generate_cutoffs(
    all_dates: list[pd.Timestamp],
    n_windows: int,
    horizon: int,
    step: int,
    min_train_size: int,
    strategy: Literal["expanding", "sliding"],
) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    """Generate cutoff dates for backtest windows."""
    cutoffs = []

    if strategy == "expanding":
        start_idx = min_train_size
        for i in range(n_windows):
            cutoff_idx = start_idx + i * step
            if cutoff_idx + horizon > len(all_dates):
                break

            cutoff_date = all_dates[cutoff_idx]
            test_dates = all_dates[cutoff_idx : cutoff_idx + horizon]
            cutoffs.append((cutoff_date, test_dates))

    elif strategy == "sliding":
        total_needed = min_train_size + (n_windows - 1) * step + horizon
        if total_needed > len(all_dates):
            n_windows = ((len(all_dates) - min_train_size - horizon) // step) + 1
            n_windows = max(n_windows, 0)

        for i in range(n_windows):
            train_end_idx = min_train_size + i * step
            cutoff_date = all_dates[train_end_idx]
            test_dates = all_dates[train_end_idx : train_end_idx + horizon]
            cutoffs.append((cutoff_date, test_dates))

    return cutoffs


def cross_validation_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    horizon: int = 1,
    gap: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate cross-validation splits with temporal validation."""
    validate_temporal_ordering(df)

    splits = []
    dates = sorted(df["ds"].unique())
    fold_size = (len(dates) - horizon) // n_splits

    for i in range(n_splits):
        split_point = (i + 1) * fold_size
        train_end = dates[split_point - 1]
        test_start_idx = split_point + gap

        if test_start_idx + horizon > len(dates):
            break

        test_dates = dates[test_start_idx : test_start_idx + horizon]
        train_df = df[df["ds"] <= train_end].copy()
        test_df = df[df["ds"].isin(test_dates)].copy()
        splits.append((train_df, test_df))

    return splits


__all__ = [
    "cross_validation_split",
    "generate_cutoffs",
    "validate_temporal_ordering",
]

