"""Window-level backtest helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsagentkit.covariates import AlignedDataset, align_covariates

if TYPE_CHECKING:
    from tsagentkit.contracts import TaskSpec
    from tsagentkit.hierarchy import HierarchyStructure, ReconciliationMethod
    from tsagentkit.series import TSDataset


def reconcile_forecast(
    forecast_df: pd.DataFrame,
    hierarchy: HierarchyStructure,
    method: str | ReconciliationMethod,
) -> pd.DataFrame:
    """Reconcile forecast to ensure hierarchy coherence."""
    from tsagentkit.hierarchy.utils import apply_reconciliation_if_needed

    return apply_reconciliation_if_needed(forecast_df, hierarchy, method)


def build_window_covariates(
    dataset: TSDataset,
    task_spec: TaskSpec,
    cutoff_date: pd.Timestamp,
    panel_for_index: pd.DataFrame,
) -> AlignedDataset | None:
    """Align covariates for a specific backtest window."""
    if dataset.panel_with_covariates is None and dataset.covariate_bundle is None:
        return None

    ds_col = task_spec.panel_contract.ds_col
    y_col = task_spec.panel_contract.y_col

    if dataset.panel_with_covariates is not None:
        panel = dataset.panel_with_covariates.copy()
    else:
        panel = panel_for_index.copy()

    if y_col in panel.columns:
        panel.loc[panel[ds_col] >= cutoff_date, y_col] = np.nan

    return align_covariates(
        panel=panel,
        task_spec=task_spec,
        covariates=dataset.covariate_bundle,
    )


__all__ = ["build_window_covariates", "reconcile_forecast"]

