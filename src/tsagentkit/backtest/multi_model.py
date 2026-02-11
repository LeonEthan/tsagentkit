"""Multi-model backtest engine for per-series model selection.

This module provides functionality to run backtests across multiple candidate
models and select the best-performing model for each series based on a
configurable metric.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .engine import rolling_backtest
from .report import BacktestReport

if TYPE_CHECKING:
    from tsagentkit.contracts import TaskSpec
    from tsagentkit.router import PlanSpec
    from tsagentkit.series import TSDataset


@dataclass(frozen=True)
class SeriesModelRanking:
    """Best model selection for a single series.

    Attributes:
        unique_id: Series identifier
        best_model: Name of the best-performing model
        best_metric_value: Metric value for the best model
        all_model_metrics: Metrics for all evaluated models
    """

    unique_id: str
    best_model: str
    best_metric_value: float
    all_model_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unique_id": self.unique_id,
            "best_model": self.best_model,
            "best_metric_value": self.best_metric_value,
            "all_model_metrics": self.all_model_metrics,
        }


@dataclass
class MultiModelBacktestReport:
    """Results from multi-model backtest with per-series selection.

    Contains backtest reports for each candidate model and the per-series
    model selection based on the specified metric.

    Attributes:
        per_model_reports: Backtest reports indexed by model name
        series_rankings: Best model selection for each series
        selection_metric: Metric used for model selection
        candidate_models: List of candidate models that were evaluated
        selection_map: Quick lookup mapping unique_id to best model
        n_windows: Number of backtest windows
        strategy: Window strategy used
    """

    per_model_reports: dict[str, BacktestReport] = field(default_factory=dict)
    series_rankings: dict[str, SeriesModelRanking] = field(default_factory=dict)
    selection_metric: str = "smape"
    candidate_models: list[str] = field(default_factory=list)
    selection_map: dict[str, str] = field(default_factory=dict)
    n_windows: int = 0
    strategy: str = "expanding"

    def get_model_for_series(self, unique_id: str) -> str | None:
        """Get the best model for a specific series.

        Args:
            unique_id: Series identifier

        Returns:
            Best model name or None if not found
        """
        return self.selection_map.get(unique_id)

    def get_model_distribution(self) -> dict[str, int]:
        """Get count of series assigned to each model.

        Returns:
            Dictionary mapping model name to count of series
        """
        distribution: dict[str, int] = defaultdict(int)
        for model_name in self.selection_map.values():
            distribution[model_name] += 1
        return dict(distribution)

    def get_aggregate_metrics(self) -> dict[str, dict[str, float]]:
        """Get aggregate metrics per model.

        Returns:
            Dictionary mapping model name to aggregate metrics
        """
        result: dict[str, dict[str, float]] = {}
        for model_name, report in self.per_model_reports.items():
            result[model_name] = report.aggregate_metrics.copy()
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "selection_metric": self.selection_metric,
            "candidate_models": self.candidate_models,
            "selection_map": self.selection_map,
            "n_windows": self.n_windows,
            "strategy": self.strategy,
            "model_distribution": self.get_model_distribution(),
            "series_rankings": {
                uid: ranking.to_dict() for uid, ranking in self.series_rankings.items()
            },
            "per_model_aggregate_metrics": self.get_aggregate_metrics(),
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "Multi-Model Backtest Report",
            f"Selection Metric: {self.selection_metric}",
            f"Candidate Models: {', '.join(self.candidate_models)}",
            f"Windows: {self.n_windows} ({self.strategy})",
            "=" * 50,
        ]

        # Model distribution
        distribution = self.get_model_distribution()
        lines.append("\nModel Distribution:")
        for model, count in sorted(distribution.items(), key=lambda x: -x[1]):
            lines.append(f"  {model}: {count} series")

        # Aggregate metrics per model
        lines.append("\nAggregate Metrics by Model:")
        agg_metrics = self.get_aggregate_metrics()
        for model, metrics in sorted(agg_metrics.items()):
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in list(metrics.items())[:3]
            )
            lines.append(f"  {model}: {metric_str}")

        return "\n".join(lines)


def multi_model_backtest(
    dataset: TSDataset,
    spec: TaskSpec,
    plan: PlanSpec,
    selection_metric: str = "smape",
    n_windows: int | None = None,
    min_train_size: int | None = None,
    step_size: int | None = None,
    fit_func: Any | None = None,
    predict_func: Any | None = None,
) -> MultiModelBacktestReport:
    """Run backtest for all candidates and select best per series.

    Executes rolling backtest for each candidate model, collects per-series
    metrics, and selects the best-performing model for each series based
    on the specified metric (lower is better).

    Args:
        dataset: TSDataset with time series data
        spec: Task specification
        plan: Execution plan with candidate models
        selection_metric: Metric for model selection (default: "smape")
        n_windows: Number of backtest windows (default: from spec)
        min_train_size: Minimum training observations (default: from spec)
        step_size: Step size between windows (default: from spec)
        fit_func: Optional custom fit function
        predict_func: Optional custom predict function

    Returns:
        MultiModelBacktestReport with per-series model selection
    """
    # Get configuration from spec if not provided
    backtest_cfg = spec.backtest
    n_windows = n_windows or backtest_cfg.n_windows
    min_train_size = min_train_size or backtest_cfg.min_train_size
    step_size = step_size if step_size is not None else backtest_cfg.step

    per_model_reports: dict[str, BacktestReport] = {}
    candidate_models = list(plan.candidate_models)

    # Run backtest for each candidate model
    for model_name in candidate_models:
        # Create a single-model plan for this candidate
        single_model_plan = plan.model_copy(update={"candidate_models": [model_name]})

        try:
            report = rolling_backtest(
                dataset=dataset,
                spec=spec,
                plan=single_model_plan,
                fit_func=fit_func,
                predict_func=predict_func,
                n_windows=n_windows,
                min_train_size=min_train_size,
                step_size=step_size,
            )
            per_model_reports[model_name] = report
        except Exception:
            # Skip models that fail during backtest
            continue

    if not per_model_reports:
        raise ValueError("All candidate models failed during backtest")

    # Select best model per series
    series_rankings = _select_best_per_series(
        per_model_reports=per_model_reports,
        selection_metric=selection_metric,
    )

    # Build selection map for quick lookup
    selection_map = {uid: ranking.best_model for uid, ranking in series_rankings.items()}

    # Get strategy from first available report
    first_report = next(iter(per_model_reports.values()))
    strategy = first_report.strategy
    actual_n_windows = first_report.n_windows

    return MultiModelBacktestReport(
        per_model_reports=per_model_reports,
        series_rankings=series_rankings,
        selection_metric=selection_metric,
        candidate_models=candidate_models,
        selection_map=selection_map,
        n_windows=actual_n_windows,
        strategy=strategy,
    )


def _select_best_per_series(
    per_model_reports: dict[str, BacktestReport],
    selection_metric: str,
) -> dict[str, SeriesModelRanking]:
    """Select the best model for each series based on the selection metric.

    Args:
        per_model_reports: Backtest reports per model
        selection_metric: Metric to use for selection (lower is better)

    Returns:
        Dictionary mapping unique_id to SeriesModelRanking
    """
    # Collect all series IDs across all models
    all_series_ids: set[str] = set()
    for report in per_model_reports.values():
        all_series_ids.update(report.series_metrics.keys())

    rankings: dict[str, SeriesModelRanking] = {}

    for unique_id in all_series_ids:
        model_metrics: dict[str, dict[str, float]] = {}
        best_model: str | None = None
        best_value: float = float("inf")

        for model_name, report in per_model_reports.items():
            if unique_id in report.series_metrics:
                metrics = report.series_metrics[unique_id].metrics
                model_metrics[model_name] = metrics.copy()

                # Get the selection metric value
                metric_value = metrics.get(selection_metric, float("inf"))

                # Handle NaN values
                if np.isnan(metric_value):
                    metric_value = float("inf")

                if metric_value < best_value:
                    best_value = metric_value
                    best_model = model_name

        # If no model had valid metrics, use the first available
        if best_model is None and model_metrics:
            best_model = next(iter(model_metrics.keys()))
            best_value = float("nan")

        if best_model is not None:
            rankings[unique_id] = SeriesModelRanking(
                unique_id=unique_id,
                best_model=best_model,
                best_metric_value=best_value,
                all_model_metrics=model_metrics,
            )

    return rankings


__all__ = [
    "SeriesModelRanking",
    "MultiModelBacktestReport",
    "multi_model_backtest",
]
