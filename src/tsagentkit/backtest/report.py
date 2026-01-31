"""Backtest report structures.

Defines data classes for backtest results and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WindowResult:
    """Results from a single backtest window.

    Attributes:
        window_index: Index of the window (0-based)
        train_start: Start date of training set
        train_end: End date of training set
        test_start: Start date of test set
        test_end: End date of test set
        metrics: Dictionary of metrics for this window
        num_series: Number of series evaluated
        num_observations: Number of observations in test set
    """

    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict[str, float] = field(default_factory=dict)
    num_series: int = 0
    num_observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_index": self.window_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "metrics": self.metrics,
            "num_series": self.num_series,
            "num_observations": self.num_observations,
        }


@dataclass(frozen=True)
class SeriesMetrics:
    """Metrics aggregated by series.

    Attributes:
        series_id: Unique identifier for the series
        metrics: Dictionary of metric name to value
        num_windows: Number of windows this series appeared in
    """

    series_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    num_windows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "series_id": self.series_id,
            "metrics": self.metrics,
            "num_windows": self.num_windows,
        }


@dataclass(frozen=True)
class BacktestReport:
    """Complete backtest results.

    Contains window-level results, aggregate metrics, and series-level
    diagnostics for a backtest run.

    Attributes:
        n_windows: Number of backtest windows
        strategy: Window strategy ("expanding" or "sliding")
        window_results: List of results per window
        aggregate_metrics: Metrics aggregated across all windows
        series_metrics: Metrics aggregated by series
        errors: List of errors encountered during backtest
        metadata: Additional backtest metadata
    """

    n_windows: int
    strategy: str
    window_results: list[WindowResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    series_metrics: dict[str, SeriesMetrics] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric_name: str) -> float:
        """Get an aggregate metric by name.

        Args:
            metric_name: Name of the metric (e.g., "wape", "smape")

        Returns:
            Metric value or NaN if not found
        """
        return self.aggregate_metrics.get(metric_name, float("nan"))

    def get_series_metric(self, series_id: str, metric_name: str) -> float:
        """Get a metric for a specific series.

        Args:
            series_id: Series identifier
            metric_name: Name of the metric

        Returns:
            Metric value or NaN if not found
        """
        if series_id not in self.series_metrics:
            return float("nan")
        return self.series_metrics[series_id].metrics.get(metric_name, float("nan"))

    def get_best_series(self, metric_name: str = "wape") -> str | None:
        """Get the series with the best performance for a metric.

        Args:
            metric_name: Metric to optimize (lower is better)

        Returns:
            Series ID with lowest metric value, or None if no data
        """
        if not self.series_metrics:
            return None

        best_id = None
        best_value = float("inf")

        for sid, sm in self.series_metrics.items():
            value = sm.metrics.get(metric_name, float("inf"))
            if value < best_value:
                best_value = value
                best_id = sid

        return best_id

    def get_worst_series(self, metric_name: str = "wape") -> str | None:
        """Get the series with the worst performance for a metric.

        Args:
            metric_name: Metric to check (higher is worse)

        Returns:
            Series ID with highest metric value, or None if no data
        """
        if not self.series_metrics:
            return None

        worst_id = None
        worst_value = float("-inf")

        for sid, sm in self.series_metrics.items():
            value = sm.metrics.get(metric_name, float("-inf"))
            if value > worst_value and not pd.isna(value):
                worst_value = value
                worst_id = sid

        return worst_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_windows": self.n_windows,
            "strategy": self.strategy,
            "aggregate_metrics": self.aggregate_metrics,
            "series_metrics": {
                k: v.to_dict() for k, v in self.series_metrics.items()
            },
            "window_results": [w.to_dict() for w in self.window_results],
            "errors": self.errors,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Summary string with key metrics
        """
        lines = [
            f"Backtest Report: {self.n_windows} windows ({self.strategy})",
            "=" * 50,
        ]

        # Aggregate metrics
        lines.append("\nAggregate Metrics:")
        for name, value in sorted(self.aggregate_metrics.items()):
            lines.append(f"  {name}: {value:.4f}")

        # Best/worst series
        best = self.get_best_series("wape")
        worst = self.get_worst_series("wape")
        if best and worst:
            lines.append(f"\nBest Series: {best}")
            lines.append(f"Worst Series: {worst}")

        # Errors
        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")

        return "\n".join(lines)
