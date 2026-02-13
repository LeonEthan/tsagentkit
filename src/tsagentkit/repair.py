"""Standalone data repair utility for tsagentkit.

Provides a ``repair()`` function that validates an input DataFrame and
applies safe, deterministic fixes for common data issues.

Usage:
    >>> from tsagentkit import repair
    >>> repaired_df, actions = repair(df)

Import layering: repair.py imports from contracts, utils, series (data_utils
layer) only — never from models or serving.
"""

from __future__ import annotations

from typing import TypedDict, cast

import pandas as pd

from tsagentkit.contracts.results import ValidationReport
from tsagentkit.contracts.task_spec import PanelContract, TaskSpec


class RepairAction(TypedDict):
    action: str
    description: str
    details: dict[str, object]


def repair(
    df: pd.DataFrame,
    spec: TaskSpec | None = None,
    *,
    panel_contract: PanelContract | None = None,
) -> tuple[pd.DataFrame, list[RepairAction]]:
    """Validate and apply safe automatic repairs to a DataFrame.

    Runs ``validate_contract`` to detect issues, then applies deterministic
    fixes that are always safe (no information loss beyond duplicate removal).

    Supported automatic fixes:
        * ``E_DS_NOT_MONOTONIC`` — sort by ``[unique_id, ds]``
        * ``E_CONTRACT_DUPLICATE_KEY`` — drop duplicates keeping last
        * Future null rows — drop rows beyond last observed ``y`` per series

    Args:
        df: Input DataFrame (not mutated; a copy is returned).
        spec: Optional TaskSpec — used to derive panel column names.
        panel_contract: Optional explicit PanelContract. If *spec* is
            provided this is ignored.

    Returns:
        A tuple of ``(repaired_df, actions)`` where *actions* is a list
        of dicts describing each repair applied.  Each dict has keys
        ``action``, ``description``, and ``details``.
    """
    from tsagentkit.contracts.schema import validate_contract
    from tsagentkit.utils.temporal import drop_future_rows

    contract = _resolve_contract(spec, panel_contract)
    uid_col = contract.unique_id_col
    ds_col = contract.ds_col
    y_col = contract.y_col

    actions: list[RepairAction] = []
    result = df.copy()

    # Step 1: Run validation to detect issues
    report = cast(
        ValidationReport,
        validate_contract(
            result,
            panel_contract=contract,
            apply_aggregation=False,
            return_data=False,
        ),
    )

    error_codes = {e.get("code", "") for e in report.errors}

    # Step 2: Fix unsorted / non-monotonic
    if "E_DS_NOT_MONOTONIC" in error_codes or not _is_sorted(result, uid_col, ds_col):
        result = result.sort_values([uid_col, ds_col]).reset_index(drop=True)
        actions.append(
            {
                "action": "sort",
                "description": f"Sorted DataFrame by ['{uid_col}', '{ds_col}']",
                "details": {"columns": [uid_col, ds_col]},
            }
        )

    # Step 3: Fix duplicate keys
    if "E_CONTRACT_DUPLICATE_KEY" in error_codes or _has_duplicates(result, uid_col, ds_col):
        n_before = len(result)
        result = result.drop_duplicates(subset=[uid_col, ds_col], keep="last")
        result = result.reset_index(drop=True)
        n_dropped = n_before - len(result)
        if n_dropped > 0:
            actions.append(
                {
                    "action": "drop_duplicates",
                    "description": f"Dropped {n_dropped} duplicate ({uid_col}, {ds_col}) rows (kept last)",
                    "details": {"rows_dropped": n_dropped},
                }
            )

    # Step 4: Drop future rows (y is null beyond last observed per series)
    result, drop_info = drop_future_rows(
        result,
        id_col=uid_col,
        ds_col=ds_col,
        y_col=y_col,
    )
    if drop_info:
        actions.append(
            {
                "action": "drop_future_rows",
                "description": "Dropped rows beyond last observed y per series",
                "details": cast(dict[str, object], drop_info),
            }
        )

    return result, actions


def _resolve_contract(
    spec: TaskSpec | None,
    panel_contract: PanelContract | None,
) -> PanelContract:
    """Resolve which PanelContract to use."""
    if spec is not None:
        return spec.panel_contract
    if panel_contract is not None:
        return panel_contract
    return PanelContract()


def _is_sorted(df: pd.DataFrame, uid_col: str, ds_col: str) -> bool:
    """Check if DataFrame is sorted by [uid_col, ds_col]."""
    if df.empty:
        return True
    sorted_df = df.sort_values([uid_col, ds_col])
    return (
        df[uid_col].tolist() == sorted_df[uid_col].tolist()
        and df[ds_col].tolist() == sorted_df[ds_col].tolist()
    )


def _has_duplicates(df: pd.DataFrame, uid_col: str, ds_col: str) -> bool:
    """Check if DataFrame has duplicate (uid_col, ds_col) pairs."""
    if df.empty:
        return False
    return df.duplicated(subset=[uid_col, ds_col]).any()


__all__ = ["repair", "RepairAction"]
