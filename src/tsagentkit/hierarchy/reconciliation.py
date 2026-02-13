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

    @classmethod
    def from_string(cls, value: str) -> ReconciliationMethod:
        """Parse method from string, defaulting to BOTTOM_UP if invalid.

        Args:
            value: String representation of the method

        Returns:
            ReconciliationMethod enum value

        Example:
            >>> ReconciliationMethod.from_string("bottom_up")
            <ReconciliationMethod.BOTTOM_UP: 'bottom_up'>
            >>> ReconciliationMethod.from_string("invalid")
            <ReconciliationMethod.BOTTOM_UP: 'bottom_up'>
        """
        mapping = {m.value: m for m in cls}
        return mapping.get(value, cls.BOTTOM_UP)


def _build_tag_indices(tags: dict[str, np.ndarray], order: list[str]) -> dict[str, np.ndarray]:
    node_to_idx = {node: idx for idx, node in enumerate(order)}
    indexed: dict[str, np.ndarray] = {}
    for key, nodes in tags.items():
        indices = [node_to_idx[n] for n in nodes if n in node_to_idx]
        if indices:
            indexed[key] = np.array(indices, dtype=int)
    return indexed


def _build_s_matrix(structure: HierarchyStructure, order: list[str]) -> np.ndarray:
    s_df = structure.to_s_df()
    s_df = s_df.set_index("unique_id").reindex(order)
    return s_df[structure.bottom_nodes].to_numpy(dtype=float)


def _get_level_key(tags: dict[str, np.ndarray], middle_level: int | str | None) -> str:
    level_keys = [k for k in tags if k.startswith("level_")]
    level_keys = sorted(level_keys, key=lambda k: int(k.split("_")[1]))
    if not level_keys:
        return next(iter(tags), "bottom")

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
    try:
        from hierarchicalforecast.methods import BottomUp, MiddleOut, MinTrace, TopDown
    except ImportError as e:
        raise ImportError(
            "hierarchicalforecast>=1.0.0 is required for reconciliation."
        ) from e

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
        if not result:
            reconciler_name = reconciler.__class__.__name__
            raise ValueError(
                f"Reconciler '{reconciler_name}' returned empty dict. "
                "This may indicate incompatible input data or a configuration issue."
            )
        if "mean" in result:
            return np.asarray(result["mean"])
        # Fallback to first array-like entry
        for _key, value in result.items():
            if isinstance(value, (np.ndarray, list)):
                return np.asarray(value)
        # No array-like values found
        reconciler_name = reconciler.__class__.__name__
        result_keys = list(result.keys())
        raise ValueError(
            f"Reconciler '{reconciler_name}' returned dict with unsupported value types. "
            f"Expected array-like values, got keys: {result_keys} with value types: "
            f"{[type(v).__name__ for v in result.values()]}. "
            "Dict must contain 'mean' key or array-like values."
        )
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

        hf_order = self.structure.node_order()
        tags = self.structure.to_tags()
        indexed_tags = _build_tag_indices(tags, hf_order)
        s_matrix = _build_s_matrix(self.structure, hf_order)
        order_to_hf = [self.structure.all_nodes.index(node) for node in hf_order]
        hf_to_order = np.argsort(order_to_hf)
        y_hat = y_hat[order_to_hf]

        y_insample = None
        y_hat_insample = None
        has_insample = False
        if fitted_values is not None and residuals is not None:
            y_hat_insample = np.asarray(fitted_values, dtype=float)[order_to_hf]
            residuals_arr = np.asarray(residuals, dtype=float)[order_to_hf]
            y_insample = y_hat_insample + residuals_arr
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
            indexed_tags,
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

    # Validate required columns exist
    missing_cols = []
    if id_col not in df.columns:
        missing_cols.append(id_col)
    if ds_col not in df.columns:
        missing_cols.append(ds_col)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in base_forecasts: {missing_cols}. "
            f"Expected columns: '{id_col}' (series identifier) and '{ds_col}' (timestamp)."
        )

    if not pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        df[ds_col] = pd.to_datetime(df[ds_col])

    value_cols = [
        c for c in df.columns if c not in {id_col, ds_col} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not value_cols:
        raise ValueError("No numeric forecast columns to reconcile.")

    try:
        from hierarchicalforecast.core import HierarchicalReconciliation
    except ImportError as e:
        raise ImportError(
            "hierarchicalforecast>=1.0.0 is required for reconciliation."
        ) from e

    tags = structure.to_tags()
    s_df = structure.to_s_df()
    reconciler = _select_reconciler(method, tags)
    engine = HierarchicalReconciliation([reconciler])

    y_hat_df = df[[id_col, ds_col] + value_cols].copy()
    reconciled = engine.reconcile(
        Y_hat_df=y_hat_df,
        S_df=s_df,
        tags=tags,
        Y_df=None,
        id_col=id_col,
        time_col=ds_col,
        target_col="y",
    )

    method_label = reconciler.__class__.__name__
    reconciled = reconciled.copy()
    for col in value_cols:
        candidate = f"{col}/{method_label}"
        if candidate in reconciled.columns:
            reconciled[col] = reconciled[candidate]
        else:
            alternatives = [c for c in reconciled.columns if c.startswith(f"{col}/")]
            if len(alternatives) == 1:
                reconciled[col] = reconciled[alternatives[0]]
    drop_cols = [c for c in reconciled.columns if "/" in c and c.split("/")[0] in value_cols]
    if drop_cols:
        reconciled = reconciled.drop(columns=drop_cols)

    return reconciled.sort_values([id_col, ds_col]).reset_index(drop=True)


__all__ = ["ReconciliationMethod", "Reconciler", "reconcile_forecasts"]
