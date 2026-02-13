"""Data bucketing for advanced router strategies.

Provides Head vs Tail and Short vs Long history bucketing for
series-specific model selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.series import SparsityProfile, TSDataset


class SeriesBucket(Enum):
    """Buckets for series classification.

    - HEAD: High volume/frequent series (top 20% by default)
    - TAIL: Low volume/infrequent series (bottom 20% by default)
    - SHORT_HISTORY: Few observations (< 30 by default)
    - LONG_HISTORY: Many observations (> 365 by default)
    """

    HEAD = "head"
    TAIL = "tail"
    SHORT_HISTORY = "short_history"
    LONG_HISTORY = "long_history"


class TSFMPolicyLike(Protocol):
    """Minimal TSFM policy shape used by bucketing recommendations."""

    mode: str
    adapters: Sequence[str]


@dataclass(frozen=True)
class BucketStatistics:
    """Statistics for a bucket.

    Attributes:
        series_count: Number of series in the bucket
        total_observations: Total observations across all series
        avg_observations: Average observations per series
        avg_value: Average value (e.g., mean sales)
        value_percentile: Percentile of total value volume
    """

    series_count: int
    total_observations: int
    avg_observations: float
    avg_value: float
    value_percentile: float


@dataclass(frozen=True)
class BucketConfig:
    """Configuration for data bucketing thresholds.

    Attributes:
        head_quantile_threshold: Quantile for HEAD bucket (default: 0.8, top 20%)
        tail_quantile_threshold: Quantile for TAIL bucket (default: 0.2, bottom 20%)
        short_history_max_obs: Max observations for SHORT_HISTORY (default: 30)
        long_history_min_obs: Min observations for LONG_HISTORY (default: 365)
        prefer_sparsity: Whether sparsity classification trumps volume (default: True)
        value_col: Column to use for volume calculations (default: "y")

    Example:
        >>> config = BucketConfig(
        ...     head_quantile_threshold=0.8,  # Top 20% by volume = HEAD
        ...     short_history_max_obs=50,     # < 50 obs = short history
        ... )
    """

    head_quantile_threshold: float = 0.8
    tail_quantile_threshold: float = 0.2
    short_history_max_obs: int = 30
    long_history_min_obs: int = 365
    prefer_sparsity: bool = True
    value_col: str = "y"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 < self.head_quantile_threshold <= 1:
            raise ValueError("head_quantile_threshold must be in (0, 1]")
        if not 0 <= self.tail_quantile_threshold < 1:
            raise ValueError("tail_quantile_threshold must be in [0, 1)")
        if self.tail_quantile_threshold >= self.head_quantile_threshold:
            raise ValueError("tail_quantile_threshold must be < head_quantile_threshold")
        if self.short_history_max_obs < 1:
            raise ValueError("short_history_max_obs must be >= 1")
        if self.long_history_min_obs <= self.short_history_max_obs:
            raise ValueError("long_history_min_obs must be > short_history_max_obs")


@dataclass(frozen=True)
class BucketProfile:
    """Bucketing profile for a dataset.

    Attributes:
        bucket_assignments: Dict mapping unique_id to set of assigned buckets
        bucket_stats: Dict mapping bucket to BucketStatistics

    Example:
        >>> profile = BucketProfile(
        ...     bucket_assignments={"A": {SeriesBucket.HEAD, SeriesBucket.LONG_HISTORY}},
        ...     bucket_stats={SeriesBucket.HEAD: stats},
        ... )
        >>> print(profile.get_bucket_for_series("A"))
        {SeriesBucket.HEAD, SeriesBucket.LONG_HISTORY}
    """

    bucket_assignments: dict[str, set[SeriesBucket]] = field(default_factory=dict)
    bucket_stats: dict[SeriesBucket, BucketStatistics] = field(default_factory=dict)

    def get_bucket_for_series(self, unique_id: str) -> set[SeriesBucket]:
        """Get all buckets assigned to a series.

        Args:
            unique_id: Series identifier

        Returns:
            Set of buckets for the series (empty if not found)
        """
        return self.bucket_assignments.get(unique_id, set())

    def get_series_in_bucket(self, bucket: SeriesBucket) -> list[str]:
        """Get all series IDs in a specific bucket.

        Args:
            bucket: Bucket to query

        Returns:
            List of unique_id values in the bucket
        """
        return [uid for uid, buckets in self.bucket_assignments.items() if bucket in buckets]

    def get_bucket_counts(self) -> dict[SeriesBucket, int]:
        """Get count of series per bucket.

        Returns:
            Dict mapping bucket to series count
        """
        counts: dict[SeriesBucket, int] = dict.fromkeys(SeriesBucket, 0)
        for buckets in self.bucket_assignments.values():
            for bucket in buckets:
                counts[bucket] = counts.get(bucket, 0) + 1
        return counts

    def summary(self) -> str:
        """Generate human-readable summary of bucketing."""
        counts = self.get_bucket_counts()
        lines = ["Bucket Profile Summary:"]
        for bucket in SeriesBucket:
            count = counts.get(bucket, 0)
            lines.append(f"  {bucket.value}: {count} series")
        return "\n".join(lines)


class DataBucketer:
    """Bucket series by volume and history characteristics.

    This class classifies series into buckets based on:
    - Volume: Head (high volume) vs Tail (low volume)
    - History: Short history vs Long history

    Example:
        >>> config = BucketConfig(
        ...     head_quantile_threshold=0.8,
        ...     short_history_max_obs=50,
        ... )
        >>> bucketer = DataBucketer(config)
        >>> profile = bucketer.create_bucket_profile(dataset)
        >>> print(profile.summary())
        Bucket Profile Summary:
          head: 20 series
          tail: 20 series
          short_history: 15 series
          long_history: 25 series
    """

    def __init__(self, config: BucketConfig | None = None):
        """Initialize the bucketer with configuration.

        Args:
            config: Bucketing configuration (uses defaults if None)
        """
        self.config = config or BucketConfig()

    def bucket_by_volume(
        self,
        df: pd.DataFrame,
        value_col: str | None = None,
    ) -> dict[str, SeriesBucket]:
        """Bucket series into HEAD/TAIL based on total value volume.

        Args:
            df: DataFrame with [unique_id, value_col] columns
            value_col: Column containing values to sum (uses config default)

        Returns:
            Dict mapping unique_id to HEAD or TAIL bucket
        """
        value_col = value_col or self.config.value_col

        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in data")

        # Compute total volume per series
        volume_by_series = df.groupby("unique_id")[value_col].sum().reset_index()
        volume_by_series.columns = ["unique_id", "total_volume"]

        # Sort by volume
        volume_by_series = volume_by_series.sort_values("total_volume")

        n_series = len(volume_by_series)
        if n_series == 0:
            return {}

        # Compute quantile positions
        volume_by_series["rank"] = range(n_series)
        volume_by_series["quantile"] = (
            volume_by_series["rank"] / (n_series - 1) if n_series > 1 else 0
        )

        # Assign buckets
        buckets = {}
        for _, row in volume_by_series.iterrows():
            uid = row["unique_id"]
            quantile = row["quantile"]

            if quantile >= self.config.head_quantile_threshold:
                buckets[uid] = SeriesBucket.HEAD
            elif quantile <= self.config.tail_quantile_threshold:
                buckets[uid] = SeriesBucket.TAIL

        return buckets

    def bucket_by_history_length(
        self,
        df: pd.DataFrame,
    ) -> dict[str, SeriesBucket]:
        """Bucket series into SHORT_HISTORY/LONG_HISTORY by observation count.

        Args:
            df: DataFrame with [unique_id] column

        Returns:
            Dict mapping unique_id to SHORT_HISTORY or LONG_HISTORY bucket
        """
        # Count observations per series
        obs_counts = df.groupby("unique_id").size().reset_index(name="n_obs")

        buckets = {}
        for _, row in obs_counts.iterrows():
            uid = row["unique_id"]
            n_obs = row["n_obs"]

            if n_obs <= self.config.short_history_max_obs:
                buckets[uid] = SeriesBucket.SHORT_HISTORY
            elif n_obs >= self.config.long_history_min_obs:
                buckets[uid] = SeriesBucket.LONG_HISTORY

        return buckets

    def create_bucket_profile(
        self,
        dataset: TSDataset,
        sparsity_profile: SparsityProfile | None = None,
    ) -> BucketProfile:
        """Create comprehensive bucket profile combining all bucketing strategies.

        A series can belong to multiple buckets (e.g., HEAD + LONG_HISTORY).

        Args:
            dataset: TSDataset to analyze
            sparsity_profile: Optional sparsity profile for sparsity-based overrides

        Returns:
            BucketProfile with assignments and statistics
        """
        # Extract DataFrame from TSDataset
        if hasattr(dataset, "df"):
            df = dataset.df
        elif hasattr(dataset, "data"):
            df = dataset.data
        else:
            df = dataset

        # Ensure required columns exist
        if "unique_id" not in df.columns:
            raise ValueError("Dataset must have 'unique_id' column")

        # Get volume buckets
        volume_buckets = self.bucket_by_volume(df)

        # Get history buckets
        history_buckets = self.bucket_by_history_length(df)

        # Combine bucket assignments
        bucket_assignments: dict[str, set[SeriesBucket]] = {}
        all_series = df["unique_id"].unique()

        for uid in all_series:
            buckets = set()

            # Add volume-based bucket if assigned
            if uid in volume_buckets:
                buckets.add(volume_buckets[uid])

            # Add history-based bucket if assigned
            if uid in history_buckets:
                buckets.add(history_buckets[uid])

            # If no buckets assigned, series is "middle" - not head/tail, not short/long
            # We still include it with empty bucket set
            bucket_assignments[uid] = buckets

        # Apply sparsity overrides if configured
        if sparsity_profile is not None and self.config.prefer_sparsity:
            bucket_assignments = self._apply_sparsity_overrides(
                bucket_assignments, sparsity_profile
            )

        # Compute statistics for each bucket
        bucket_stats = self._compute_bucket_stats(df, bucket_assignments)

        return BucketProfile(
            bucket_assignments=bucket_assignments,
            bucket_stats=bucket_stats,
        )

    def _apply_sparsity_overrides(
        self,
        bucket_assignments: dict[str, set[SeriesBucket]],
        sparsity_profile: SparsityProfile,
    ) -> dict[str, set[SeriesBucket]]:
        """Apply sparsity-based overrides to bucket assignments.

        Intermittent and sparse series are treated as TAIL regardless of volume.
        Cold-start series are treated as SHORT_HISTORY.

        Args:
            bucket_assignments: Current bucket assignments
            sparsity_profile: Sparsity profile from series module

        Returns:
            Updated bucket assignments
        """
        updated = dict(bucket_assignments)

        for uid in updated:
            classification = sparsity_profile.get_classification(uid)

            if classification.value in ("intermittent", "sparse"):
                # Intermittent/sparse series -> TAIL
                updated[uid] = updated[uid] | {SeriesBucket.TAIL}
                # Remove HEAD if present
                updated[uid] = updated[uid] - {SeriesBucket.HEAD}

            if classification.value == "cold_start":
                # Cold start -> SHORT_HISTORY
                updated[uid] = updated[uid] | {SeriesBucket.SHORT_HISTORY}
                # Remove LONG_HISTORY if present
                updated[uid] = updated[uid] - {SeriesBucket.LONG_HISTORY}

        return updated

    def _compute_bucket_stats(
        self,
        df: pd.DataFrame,
        bucket_assignments: dict[str, set[SeriesBucket]],
    ) -> dict[SeriesBucket, BucketStatistics]:
        """Compute statistics for each bucket.

        Args:
            df: Source DataFrame
            bucket_assignments: Bucket assignments per series

        Returns:
            Dict mapping bucket to BucketStatistics
        """
        stats = {}
        value_col = self.config.value_col

        # Compute per-series metrics
        series_metrics = (
            df.groupby("unique_id")
            .agg(
                {
                    value_col: ["sum", "mean", "count"],
                }
            )
            .reset_index()
        )
        series_metrics.columns = ["unique_id", "total_value", "avg_value", "n_obs"]

        # Total volume across all series
        total_volume = series_metrics["total_value"].sum()

        for bucket in SeriesBucket:
            series_in_bucket = [
                uid for uid, buckets in bucket_assignments.items() if bucket in buckets
            ]

            if not series_in_bucket:
                stats[bucket] = BucketStatistics(
                    series_count=0,
                    total_observations=0,
                    avg_observations=0.0,
                    avg_value=0.0,
                    value_percentile=0.0,
                )
                continue

            bucket_metrics = series_metrics[series_metrics["unique_id"].isin(series_in_bucket)]

            bucket_volume = bucket_metrics["total_value"].sum()
            value_percentile = bucket_volume / total_volume if total_volume > 0 else 0.0

            stats[bucket] = BucketStatistics(
                series_count=len(series_in_bucket),
                total_observations=int(bucket_metrics["n_obs"].sum()),
                avg_observations=float(bucket_metrics["n_obs"].mean()),
                avg_value=float(bucket_metrics["avg_value"].mean()),
                value_percentile=float(value_percentile),
            )

        return stats

    def get_model_for_bucket(
        self,
        bucket: SeriesBucket,
        sparsity_class: str | None = None,
        tsfm_policy: TSFMPolicyLike | None = None,
    ) -> str:
        """Get recommended model for a given bucket.

        Model recommendations:
        - HEAD + LONG_HISTORY: TSFM or sophisticated model
        - HEAD + SHORT_HISTORY: Robust local model
        - TAIL: Simple baseline (SeasonalNaive, HistoricAverage)
        - INTERMITTENT: Croston or ADIDA (via statsforecast)

        Args:
            bucket: Bucket to get recommendation for
            sparsity_class: Optional sparsity classification
            tsfm_policy: Optional TSFMPolicy instance. When provided and
                TSFM is not disabled, HEAD bucket returns the first
                available TSFM adapter instead of SeasonalNaive.

        Returns:
            Recommended model name
        """
        if sparsity_class == "intermittent":
            return "Croston"  # or "ADIDA"

        if bucket == SeriesBucket.HEAD:
            if tsfm_policy is not None:
                mode = getattr(tsfm_policy, "mode", "disabled")
                adapters = getattr(tsfm_policy, "adapters", [])
                if mode != "disabled" and adapters:
                    return f"tsfm-{adapters[0]}"
            return "SeasonalNaive"
        elif bucket == SeriesBucket.TAIL or bucket == SeriesBucket.SHORT_HISTORY:
            return "HistoricAverage"
        elif bucket == SeriesBucket.LONG_HISTORY:
            return "SeasonalNaive"
        else:
            return "SeasonalNaive"  # Default

    def get_bucket_specific_plan_config(
        self,
        bucket: SeriesBucket,
    ) -> dict:
        """Get bucket-specific configuration for model training.

        Args:
            bucket: Bucket to get config for

        Returns:
            Dict with bucket-specific configuration
        """
        configs = {
            SeriesBucket.HEAD: {
                "model_complexity": "high",
                "hyperparameter_tuning": True,
                "ensemble_size": 5,
            },
            SeriesBucket.TAIL: {
                "model_complexity": "low",
                "hyperparameter_tuning": False,
                "ensemble_size": 1,
            },
            SeriesBucket.SHORT_HISTORY: {
                "min_observations": 10,
                "use_global_model": True,
            },
            SeriesBucket.LONG_HISTORY: {
                "min_observations": 100,
                "use_global_model": False,
            },
        }

        return configs.get(bucket, {})
