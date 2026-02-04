"""Hierarchical time series forecasting and reconciliation.

This module provides tools for working with hierarchical time series data,
including structure definition, aggregation matrix operations, and multiple
reconciliation methods (bottom-up, top-down, middle-out, OLS, MinT).

Example:
    >>> from tsagentkit.hierarchy import HierarchyStructure, Reconciler
    >>> from tsagentkit.hierarchy import ReconciliationMethod
    >>>
    >>> # Define hierarchy
    >>> structure = HierarchyStructure.from_dataframe(
    ...     df=sales_data,
    ...     hierarchy_columns=["region", "state", "store"]
    ... )
    >>>
    >>> # Reconcile forecasts
    >>> reconciler = Reconciler(ReconciliationMethod.MIN_TRACE, structure)
    >>> reconciled = reconciler.reconcile(base_forecasts, residuals=residuals)
"""

from __future__ import annotations

from .evaluator import CoherenceViolation, HierarchyEvaluationReport, HierarchyEvaluator
from .reconciliation import Reconciler, ReconciliationMethod, reconcile_forecasts
from .structure import HierarchyStructure

__all__ = [
    # Structure
    "HierarchyStructure",
    # Reconciliation
    "Reconciler",
    "ReconciliationMethod",
    "reconcile_forecasts",
    # Evaluation
    "HierarchyEvaluator",
    "HierarchyEvaluationReport",
    "CoherenceViolation",
]
