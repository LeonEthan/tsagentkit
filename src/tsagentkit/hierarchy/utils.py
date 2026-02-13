"""Hierarchy utility functions.

Provides common helper functions for hierarchical time series operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.hierarchy import HierarchyStructure, ReconciliationMethod


def apply_reconciliation_if_needed(
    forecast: pd.DataFrame,
    hierarchy: HierarchyStructure | None,
    method: str | ReconciliationMethod = "bottom_up",
) -> pd.DataFrame:
    """Apply hierarchical reconciliation if hierarchy is provided.

    This is a convenience function that checks if reconciliation should be
    applied and delegates to the appropriate reconciliation method.

    Args:
        forecast: Forecast DataFrame with columns [unique_id, ds, yhat, ...]
        hierarchy: Hierarchy structure, or None to skip reconciliation
        method: Reconciliation method name or enum (default: "bottom_up")

    Returns:
        Reconciled forecast DataFrame if hierarchy provided, else original

    Example:
        >>> from tsagentkit.hierarchy import HierarchyStructure
        >>> from tsagentkit.hierarchy.utils import apply_reconciliation_if_needed
        >>>
        >>> # Assuming forecast_df and hierarchy_structure are defined
        >>> reconciled = apply_reconciliation_if_needed(
        ...     forecast=forecast_df,
        ...     hierarchy=hierarchy_structure,
        ...     method="bottom_up"
        ... )
    """
    if hierarchy is None:
        return forecast

    from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

    # Convert method string to enum if needed
    if isinstance(method, str):
        method = ReconciliationMethod.from_string(method)

    return reconcile_forecasts(
        base_forecasts=forecast,
        structure=hierarchy,
        method=method,
    )
