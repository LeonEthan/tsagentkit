"""Data validation schemas (compat wrapper).

This module preserves the stable API while keeping contracts free of
non-stdlib dependencies. The implementation lives in tsagentkit.series.validation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tsagentkit.contracts.results import ValidationReport
    from tsagentkit.contracts.task_spec import PanelContract


def _impl():
    return import_module("tsagentkit.series.validation")


def normalize_panel_columns(
    df: Any,
    contract: "PanelContract",
) -> tuple[Any, dict[str, str] | None]:
    """Normalize panel columns to the canonical contract names."""
    return _impl().normalize_panel_columns(df, contract)


def validate_contract(
    data: Any,
    panel_contract: "PanelContract | None" = None,
    apply_aggregation: bool = False,
    return_data: bool = False,
) -> "ValidationReport | tuple[ValidationReport, Any]":
    """Validate input data against the required schema."""
    return _impl().validate_contract(
        data,
        panel_contract=panel_contract,
        apply_aggregation=apply_aggregation,
        return_data=return_data,
    )


__all__ = ["validate_contract", "normalize_panel_columns"]
