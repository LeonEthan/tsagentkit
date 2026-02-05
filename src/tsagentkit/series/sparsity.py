"""Sparsity profiling for time series.

Identifies intermittent, cold-start, and sparse series for the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class SparsityClass(StrEnum):
    """Classification of series sparsity patterns."""

    REGULAR = "regular"
    """Regular series with consistent observations."""

    INTERMITTENT = "intermittent"
    """Series with many zero values (intermittent demand)."""

    SPARSE = "sparse"
    """Series with irregular gaps in observations."""

    COLD_START = "cold_start"
    """Series with very few observations (new series)."""


@dataclass(frozen=True)
class SparsityProfile:
    """Sparsity profile for a time series dataset.

    Contains classification and metrics for each series in the dataset,
    used by the router for model selection.

    Attributes:
        series_profiles: Dict mapping unique_id to profile metrics
        dataset_metrics: Aggregate metrics across all series
    """

    series_profiles: dict[str, dict]
    dataset_metrics: dict

    def get_classification(self, unique_id: str) -> SparsityClass:
        """Get the sparsity classification for a series.

        Args:
            unique_id: The series identifier

        Returns:
            SparsityClass enum value
        """
        profile = self.series_profiles.get(unique_id, {})
        return SparsityClass(profile.get("classification", "regular"))

    def get_series_by_class(self, cls: SparsityClass) -> list[str]:
        """Get all series IDs with a given classification.

        Args:
            cls: Sparsity class to filter by

        Returns:
            List of unique_id values
        """
        return [
            uid
            for uid, profile in self.series_profiles.items()
            if profile.get("classification") == cls.value
        ]

    def has_intermittent(self) -> bool:
        """Check if dataset has any intermittent series."""
        return any(
            p.get("classification") == SparsityClass.INTERMITTENT.value
            for p in self.series_profiles.values()
        )

    def has_cold_start(self) -> bool:
        """Check if dataset has any cold-start series."""
        return any(
            p.get("classification") == SparsityClass.COLD_START.value
            for p in self.series_profiles.values()
        )


def compute_sparsity_profile(
    df: pd.DataFrame,
    min_observations: int = 10,
    zero_threshold: float = 0.3,
    gap_threshold: float = 0.2,
) -> SparsityProfile:
    """Compute sparsity profile for a dataset.

    Analyzes each series to classify as regular, intermittent, sparse,
    or cold-start based on observation patterns.

    Args:
        df: DataFrame with columns [unique_id, ds, y]
        min_observations: Minimum observations for non-cold-start (default: 10)
        zero_threshold: Threshold for zero ratio to be intermittent (default: 0.3)
        gap_threshold: Threshold for gap ratio to be sparse (default: 0.2)

    Returns:
        SparsityProfile with classifications and metrics
    """
    series_profiles: dict[str, dict] = {}

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid].sort_values("ds")
        profile = _analyze_series(
            series,
            min_observations=min_observations,
            zero_threshold=zero_threshold,
            gap_threshold=gap_threshold,
        )
        series_profiles[uid] = profile

    # Compute dataset-level metrics
    dataset_metrics = _compute_dataset_metrics(series_profiles)

    return SparsityProfile(
        series_profiles=series_profiles,
        dataset_metrics=dataset_metrics,
    )


def _analyze_series(
    series: pd.DataFrame,
    min_observations: int,
    zero_threshold: float,
    gap_threshold: float,
) -> dict:
    """Analyze a single series for sparsity patterns.

    Args:
        series: DataFrame for a single series
        min_observations: Minimum observations threshold
        zero_threshold: Zero ratio threshold for intermittent
        gap_threshold: Gap ratio threshold for sparse

    Returns:
        Dictionary with classification and metrics
    """
    y = series["y"].values
    n = len(y)

    # Basic metrics
    metrics = {
        "n_observations": n,
        "zero_ratio": float(np.mean(y == 0)) if len(y) > 0 else 0.0,
        "missing_ratio": float(series["y"].isna().mean()),
    }

    # Compute gap metrics if we have datetime
    if n > 1 and pd.api.types.is_datetime64_any_dtype(series["ds"]):
        ds = pd.to_datetime(series["ds"])
        time_diffs = ds.diff().dropna()

        if len(time_diffs) > 0:
            # Most common interval
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                expected_interval = mode_diff.iloc[0]
                # Count gaps (intervals significantly larger than expected)
                gap_count = int((time_diffs > expected_interval * 1.5).sum())
                metrics["gap_count"] = gap_count
                metrics["gap_ratio"] = gap_count / len(time_diffs) if len(time_diffs) > 0 else 0.0
            else:
                metrics["gap_count"] = 0
                metrics["gap_ratio"] = 0.0
        else:
            metrics["gap_count"] = 0
            metrics["gap_ratio"] = 0.0
    else:
        metrics["gap_count"] = 0
        metrics["gap_ratio"] = 0.0

    # Classification logic
    classification = _classify_series(
        metrics, n, min_observations, zero_threshold, gap_threshold
    )
    metrics["classification"] = classification

    return metrics


def _classify_series(
    metrics: dict,
    n: int,
    min_observations: int,
    zero_threshold: float,
    gap_threshold: float,
) -> str:
    """Classify a series based on its metrics.

    Classification order matters:
    1. Cold start (too few observations)
    2. Intermittent (many zeros)
    3. Sparse (many gaps)
    4. Regular (default)

    Args:
        metrics: Series metrics dictionary
        n: Number of observations
        min_observations: Threshold for cold-start
        zero_threshold: Threshold for intermittent
        gap_threshold: Threshold for sparse

    Returns:
        Classification string
    """
    # Cold start: too few observations
    if n < min_observations:
        return SparsityClass.COLD_START.value

    # Intermittent: many zero values
    if metrics.get("zero_ratio", 0) > zero_threshold:
        return SparsityClass.INTERMITTENT.value

    # Sparse: significant gaps
    if metrics.get("gap_ratio", 0) > gap_threshold:
        return SparsityClass.SPARSE.value

    # Default: regular
    return SparsityClass.REGULAR.value


def _compute_dataset_metrics(series_profiles: dict[str, dict]) -> dict:
    """Compute aggregate metrics across all series.

    Args:
        series_profiles: Dict of series profiles

    Returns:
        Dictionary of dataset-level metrics
    """
    if not series_profiles:
        return {
            "total_series": 0,
            "classification_counts": {},
            "avg_observations": 0.0,
        }

    # Count classifications
    class_counts: dict[str, int] = {}
    for profile in series_profiles.values():
        cls = profile.get("classification", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Average observations
    avg_obs = sum(p.get("n_observations", 0) for p in series_profiles.values()) / len(
        series_profiles
    )

    return {
        "total_series": len(series_profiles),
        "classification_counts": class_counts,
        "avg_observations": avg_obs,
    }
