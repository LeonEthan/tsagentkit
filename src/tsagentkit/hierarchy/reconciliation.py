"""Hierarchical forecast reconciliation (adapter to hierarchicalforecast)."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .structure import HierarchyStructure


class ReconciliationMethod(Enum):
    """Available reconciliation methods."""

    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    MIDDLE_OUT = "middle_out"
    OLS = "ols"
    WLS = "wls"
    MIN_TRACE = "min_trace"


def _build_hf_order(structure: HierarchyStructure) -> list[str]:
    """Return node order with aggregates first and bottom nodes last."""
    order: list[str] = []
    for level in range(structure.get_num_levels()):
        order.extend(structure.get_nodes_at_level(level))
    bottom = [n for n in structure.bottom_nodes if n in order]
    order = [n for n in order if n not in bottom] + bottom
    return order


def _build_tags(structure: HierarchyStructure, hf_order: list[str]) -> dict[str, np.ndarray]:
    """Build tags mapping for hierarchicalforecast using row indices."""
    node_to_idx = {node: idx for idx, node in enumerate(hf_order)}
    tags: dict[str, np.ndarray] = {}
    for level in range(structure.get_num_levels()):
        nodes = structure.get_nodes_at_level(level)
        indices = [node_to_idx[n] for n in nodes if n in node_to_idx]
        if indices:
            tags[f"level_{level}"] = np.array(indices, dtype=int)
    bottom_indices = [node_to_idx[n] for n in structure.bottom_nodes if n in node_to_idx]
    tags["bottom"] = np.array(bottom_indices, dtype=int)
    return tags


def _build_s_matrix(structure: HierarchyStructure, hf_order: list[str]) -> np.ndarray:
    row_indices = [structure.all_nodes.index(node) for node in hf_order]
    return np.asarray(structure.s_matrix, dtype=float)[row_indices]


def _get_level_key(tags: dict[str, np.ndarray], middle_level: int | str | None) -> str:
    level_keys = [k for k in tags.keys() if k.startswith("level_")]
    level_keys = sorted(level_keys, key=lambda k: int(k.split("_")[1]))
    if not level_keys:
        return "bottom"

    if isinstance(middle_level, str) and middle_level in tags:
        return middle_level
    if isinstance(middle_level, int):
        idx = min(max(middle_level, 0), len(level_keys) - 1)
        return level_keys[idx]

    if len(level_keys) >= 3:
        return level_keys[len(level_keys) // 2]
    if len(level_keys) == 2:
        return level_keys[1]
    return level_keys[0]


def _select_reconciler(
    method: ReconciliationMethod,
    tags: dict[str, np.ndarray],
    middle_level: int | str | None = None,
    has_insample: bool = False,
) -> Any:
    from hierarchicalforecast.methods import BottomUp, MiddleOut, MinTrace, TopDown

    if method == ReconciliationMethod.BOTTOM_UP:
        return BottomUp()
    if method == ReconciliationMethod.TOP_DOWN:
        return TopDown(method="forecast_proportions")
    if method == ReconciliationMethod.MIDDLE_OUT:
        return MiddleOut(
            middle_level=_get_level_key(tags, middle_level),
            top_down_method="forecast_proportions",
        )
    if method == ReconciliationMethod.OLS:
        return MinTrace(method="ols")
    if method == ReconciliationMethod.WLS:
        return MinTrace(method="wls_struct")
    if method == ReconciliationMethod.MIN_TRACE:
        return MinTrace(method="mint_shrink" if has_insample else "ols")
    return MinTrace(method="ols")


def _apply_reconciler(
    reconciler: Any,
    s_matrix: np.ndarray,
    y_hat: np.ndarray,
    tags: dict[str, np.ndarray],
    y_insample: np.ndarray | None = None,
    y_hat_insample: np.ndarray | None = None,
) -> np.ndarray:
    result = reconciler.fit_predict(
        S=s_matrix,
        y_hat=y_hat,
        tags=tags,
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
    )
    if isinstance(result, dict):
        if "mean" in result:
            return np.asarray(result["mean"])
        # Fallback to first array-like entry
        for value in result.values():
            if isinstance(value, (np.ndarray, list)):
                return np.asarray(value)
        raise ValueError("Unsupported reconciliation output format.")
    return np.asarray(result)


class Reconciler:
    """Hierarchical forecast reconciliation engine (adapter)."""

    def __init__(
        self,
        method: ReconciliationMethod,
        structure: HierarchyStructure,
    ) -> None:
        self.method = method
        self.structure = structure

    def reconcile(
        self,
        base_forecasts: np.ndarray,
        fitted_values: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Reconcile base forecasts to be hierarchy-consistent."""
        y_hat = np.asarray(base_forecasts, dtype=float)
        was_1d = y_hat.ndim == 1
        if was_1d:
            y_hat = y_hat[:, None]

        hf_order = _build_hf_order(self.structure)
        tags = _build_tags(self.structure, hf_order)
        s_matrix = _build_s_matrix(self.structure, hf_order)
        order_to_hf = [self.structure.all_nodes.index(node) for node in hf_order]
        hf_to_order = np.argsort(order_to_hf)
        y_hat = y_hat[order_to_hf]

        y_insample = None
        y_hat_insample = None
        has_insample = False
        if fitted_values is not None and residuals is not None:
            y_hat_insample = np.asarray(fitted_values, dtype=float)
            y_insample = y_hat_insample + np.asarray(residuals, dtype=float)
            has_insample = True

        reconciler = _select_reconciler(
            self.method,
            tags,
            middle_level=kwargs.get("middle_level"),
            has_insample=has_insample,
        )
        reconciled = _apply_reconciler(
            reconciler,
            s_matrix,
            y_hat,
            tags,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
        )

        reconciled = reconciled[hf_to_order]
        if was_1d:
            return reconciled[:, 0]
        return reconciled


def reconcile_forecasts(
    base_forecasts: pd.DataFrame,
    structure: HierarchyStructure,
    method: ReconciliationMethod | str = ReconciliationMethod.BOTTOM_UP,
) -> pd.DataFrame:
    """Reconcile forecast DataFrame to ensure hierarchy coherence."""
    if isinstance(method, str):
        method = ReconciliationMethod(method)

    df = base_forecasts.copy()
    id_col = "unique_id"
    ds_col = "ds"
    if not pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        df[ds_col] = pd.to_datetime(df[ds_col])

    value_cols = [
        c
        for c in df.columns
        if c not in {id_col, ds_col} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not value_cols:
        raise ValueError("No numeric forecast columns to reconcile.")

    ds_values = sorted(pd.to_datetime(df[ds_col]).unique())
    hf_order = _build_hf_order(structure)
    tags = _build_tags(structure, hf_order)
    s_matrix = _build_s_matrix(structure, hf_order)
    reconciler = _select_reconciler(method, tags)

    reconciled_frames: list[pd.DataFrame] = []
    for col in value_cols:
        pivot = (
            df.pivot_table(index=id_col, columns=ds_col, values=col, aggfunc="mean")
            .reindex(index=hf_order, columns=ds_values)
        )
        if pivot.isna().any().any():
            raise ValueError(f"Missing forecasts for reconciliation in column '{col}'.")
        y_hat = pivot.to_numpy(dtype=float)
        reconciled = _apply_reconciler(reconciler, s_matrix, y_hat, tags)
        rec_df = (
            pd.DataFrame(reconciled, index=hf_order, columns=ds_values)
            .reset_index()
            .melt(id_vars="index", var_name=ds_col, value_name=col)
            .rename(columns={"index": id_col})
        )
        reconciled_frames.append(rec_df)

    # Merge reconciled columns back together
    result = reconciled_frames[0]
    for frame in reconciled_frames[1:]:
        result = result.merge(frame, on=[id_col, ds_col], how="left")

    return result.sort_values([id_col, ds_col]).reset_index(drop=True)


__all__ = ["ReconciliationMethod", "Reconciler", "reconcile_forecasts"]
