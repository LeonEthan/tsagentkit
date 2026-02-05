"""Hierarchy-aware evaluation metrics and coherence checking.

Provides tools to evaluate hierarchical forecast quality and detect
coherence violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .structure import HierarchyStructure


@dataclass(frozen=True)
class CoherenceViolation:
    """Single coherence violation record.

    Records when forecasts violate the hierarchical aggregation constraint
    (i.e., children don't sum to parent).

    Attributes:
        parent_node: Name of the parent node
        child_nodes: List of child node names
        expected_value: Expected value (sum of children)
        actual_value: Actual value (parent forecast)
        difference: Absolute difference between expected and actual
        timestamp: Timestamp of the violation
    """

    parent_node: str
    child_nodes: list[str]
    expected_value: float
    actual_value: float
    difference: float
    timestamp: str


@dataclass(frozen=True)
class HierarchyEvaluationReport:
    """Evaluation report for hierarchical forecasts.

    Contains metrics and diagnostics for hierarchical forecast quality.

    Attributes:
        level_metrics: Metrics aggregated by hierarchy level
        coherence_violations: List of coherence violations found
        coherence_score: Overall coherence score (0-1, higher is better)
        reconciliation_improvement: Improvement vs base forecasts
        total_violations: Total number of violations
        violation_rate: Proportion of forecasts with violations
    """

    level_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    coherence_violations: list[CoherenceViolation] = field(default_factory=list)
    coherence_score: float = 0.0
    reconciliation_improvement: dict[str, float] = field(default_factory=dict)
    total_violations: int = 0
    violation_rate: float = 0.0

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "level_metrics": self.level_metrics,
            "coherence_violations": [
                {
                    "parent_node": v.parent_node,
                    "child_nodes": v.child_nodes,
                    "expected_value": v.expected_value,
                    "actual_value": v.actual_value,
                    "difference": v.difference,
                    "timestamp": v.timestamp,
                }
                for v in self.coherence_violations
            ],
            "coherence_score": self.coherence_score,
            "reconciliation_improvement": self.reconciliation_improvement,
            "total_violations": self.total_violations,
            "violation_rate": self.violation_rate,
        }


class HierarchyEvaluator:
    """Evaluate hierarchical forecast quality and coherence.

    Provides methods to compute hierarchical metrics and detect
    coherence violations.

    Example:
        >>> evaluator = HierarchyEvaluator(hierarchy_structure)
        >>> report = evaluator.evaluate(forecasts, actuals)
        >>> print(f"Coherence score: {report.coherence_score:.3f}")
        Coherence score: 0.998
    """

    def __init__(self, structure: HierarchyStructure):
        """Initialize evaluator.

        Args:
            structure: Hierarchy structure
        """
        self.structure = structure

    def evaluate(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame | None = None,
        tolerance: float = 1e-6,
    ) -> HierarchyEvaluationReport:
        """Evaluate hierarchical forecasts.

        Computes standard forecast metrics per level and checks coherence.

        Args:
            forecasts: Forecast DataFrame with columns [unique_id, ds, yhat]
            actuals: Optional actual values for accuracy metrics
            tolerance: Tolerance for coherence violations

        Returns:
            HierarchyEvaluationReport with metrics and violations
        """
        # Compute per-level metrics if actuals provided
        level_metrics = {}
        if actuals is not None:
            level_metrics = self._compute_level_metrics(forecasts, actuals)

        # Detect coherence violations
        violations = self._detect_violations(forecasts, tolerance)

        # Compute coherence score
        coherence_score = self._compute_coherence_score(forecasts, tolerance)

        # Compute violation rate
        total_checks = self._count_total_checks(forecasts)
        violation_rate = len(violations) / max(total_checks, 1)

        return HierarchyEvaluationReport(
            level_metrics=level_metrics,
            coherence_violations=violations,
            coherence_score=coherence_score,
            total_violations=len(violations),
            violation_rate=violation_rate,
        )

    def _compute_level_metrics(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> dict[int, dict[str, float]]:
        """Compute metrics aggregated by hierarchy level.

        Args:
            forecasts: Forecast DataFrame
            actuals: Actual values DataFrame

        Returns:
            Dictionary mapping level to metrics dict
        """
        metrics_by_level: dict[int, dict[str, list[float]]] = {}

        # Merge forecasts with actuals
        merged = forecasts.merge(
            actuals,
            on=["unique_id", "ds"],
            suffixes=("_forecast", "_actual"),
        )

        # Compute metrics per series
        for _, row in merged.iterrows():
            uid = row["unique_id"]
            if uid not in self.structure.all_nodes:
                continue

            level = self.structure.get_level(uid)
            if level not in metrics_by_level:
                metrics_by_level[level] = {"mae": [], "mape": [], "rmse": []}

            actual = row.get("y_actual", row.get("y", 0))
            forecast = row["yhat"]
            error = forecast - actual

            metrics_by_level[level]["mae"].append(abs(error))
            metrics_by_level[level]["rmse"].append(error ** 2)
            if actual != 0:
                metrics_by_level[level]["mape"].append(abs(error / actual) * 100)

        # Aggregate
        result = {}
        for level, metrics in metrics_by_level.items():
            result[level] = {
                "mae": np.mean(metrics["mae"]) if metrics["mae"] else 0,
                "rmse": np.sqrt(np.mean(metrics["rmse"])) if metrics["rmse"] else 0,
                "mape": np.mean(metrics["mape"]) if metrics["mape"] else 0,
                "count": len(metrics["mae"]),
            }

        return result

    def _detect_violations(
        self,
        forecasts: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> list[CoherenceViolation]:
        """Detect where forecasts violate hierarchical coherence.

        A coherence violation occurs when the sum of child forecasts
        doesn't equal the parent forecast (within tolerance).

        Args:
            forecasts: Forecast DataFrame
            tolerance: Numerical tolerance for violations

        Returns:
            List of coherence violations
        """
        violations = []

        # Pivot to wide format for easier computation
        pivot = forecasts.pivot(index="unique_id", columns="ds", values="yhat")

        for parent, children in self.structure.aggregation_graph.items():
            if parent not in pivot.index:
                continue

            parent_forecast = pivot.loc[parent]

            # Sum children forecasts
            available_children = [c for c in children if c in pivot.index]
            if not available_children:
                continue

            children_sum = pivot.loc[available_children].sum()

            # Check for violations at each time point
            for ds in parent_forecast.index:
                parent_val = parent_forecast[ds]
                children_val = children_sum[ds]
                diff = abs(parent_val - children_val)

                if diff > tolerance:
                    violations.append(
                        CoherenceViolation(
                            parent_node=parent,
                            child_nodes=available_children,
                            expected_value=float(children_val),
                            actual_value=float(parent_val),
                            difference=float(diff),
                            timestamp=str(ds),
                        )
                    )

        return violations

    def _compute_coherence_score(
        self,
        forecasts: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> float:
        """Compute overall coherence score.

        Score is 1.0 if perfectly coherent, decreases with violations.

        Args:
            forecasts: Forecast DataFrame
            tolerance: Numerical tolerance

        Returns:
            Coherence score between 0 and 1
        """
        pivot = forecasts.pivot(index="unique_id", columns="ds", values="yhat")

        total_abs_sum = 0.0
        total_violation = 0.0

        for parent, children in self.structure.aggregation_graph.items():
            if parent not in pivot.index:
                continue

            parent_forecast = pivot.loc[parent]
            available_children = [c for c in children if c in pivot.index]

            if not available_children:
                continue

            children_sum = pivot.loc[available_children].sum()

            for ds in parent_forecast.index:
                parent_val = parent_forecast[ds]
                children_val = children_sum[ds]

                total_abs_sum += abs(parent_val)
                violation = abs(parent_val - children_val)

                if violation > tolerance:
                    total_violation += violation

        if total_abs_sum == 0:
            return 1.0

        # Score decreases as violation magnitude increases
        return max(0.0, 1.0 - (total_violation / total_abs_sum))

    def _count_total_checks(self, forecasts: pd.DataFrame) -> int:
        """Count total number of coherence checks performed.

        Args:
            forecasts: Forecast DataFrame

        Returns:
            Total number of parent-timestamp pairs checked
        """
        pivot = forecasts.pivot(index="unique_id", columns="ds", values="yhat")

        count = 0
        for parent, children in self.structure.aggregation_graph.items():
            if parent not in pivot.index:
                continue

            available_children = [c for c in children if c in pivot.index]
            if available_children:
                count += len(pivot.columns)

        return count

    def compute_improvement(
        self,
        base_forecasts: pd.DataFrame,
        reconciled_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute improvement of reconciled vs base forecasts.

        Args:
            base_forecasts: Base (unreconciled) forecasts
            reconciled_forecasts: Reconciled forecasts
            actuals: Actual values

        Returns:
            Dictionary with improvement metrics
        """
        base_metrics = self._compute_overall_metrics(base_forecasts, actuals)
        reconciled_metrics = self._compute_overall_metrics(
            reconciled_forecasts, actuals
        )

        improvement = {}
        for metric in ["mae", "rmse", "mape"]:
            if metric in base_metrics and base_metrics[metric] > 0:
                improvement[metric] = (
                    (base_metrics[metric] - reconciled_metrics[metric])
                    / base_metrics[metric]
                ) * 100
            else:
                improvement[metric] = 0.0

        return improvement

    def _compute_overall_metrics(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute overall metrics for forecasts.

        Args:
            forecasts: Forecast DataFrame
            actuals: Actual values DataFrame

        Returns:
            Dictionary of metrics
        """
        merged = forecasts.merge(
            actuals,
            on=["unique_id", "ds"],
            suffixes=("_forecast", "_actual"),
        )

        if len(merged) == 0:
            return {"mae": 0, "rmse": 0, "mape": 0}

        actual_col = "y_actual" if "y_actual" in merged.columns else "y"
        errors = merged["yhat"] - merged[actual_col]

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # Compute MAPE only for non-zero actuals
        non_zero_mask = merged[actual_col] != 0
        if non_zero_mask.any():
            mape = np.mean(
                np.abs(errors[non_zero_mask] / merged.loc[non_zero_mask, actual_col])
            ) * 100
        else:
            mape = 0.0

        return {"mae": mae, "rmse": rmse, "mape": mape}
