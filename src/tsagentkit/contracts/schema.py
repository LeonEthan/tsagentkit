"""Data validation schemas (compat wrapper).

This module preserves the stable API while keeping contracts free of
non-stdlib dependencies. The implementation lives in tsagentkit.series.validation.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from tsagentkit.contracts.results import ValidationReport
    from tsagentkit.contracts.task_spec import PanelContract


def _impl() -> ModuleType:
    return import_module("tsagentkit.series.validation")


def normalize_panel_columns(
    df: object,
    contract: PanelContract,
) -> tuple[object, dict[str, str] | None]:
    """Normalize panel columns to the canonical contract names."""
    return _impl().normalize_panel_columns(df, contract)


@overload
def validate_contract(
    data: object,
    panel_contract: PanelContract | None = None,
    apply_aggregation: bool = False,
    return_data: Literal[False] = False,
) -> ValidationReport: ...


@overload
def validate_contract(
    data: object,
    panel_contract: PanelContract | None = None,
    apply_aggregation: bool = False,
    return_data: Literal[True] = True,
) -> tuple[ValidationReport, object]: ...


def validate_contract(
    data: object,
    panel_contract: PanelContract | None = None,
    apply_aggregation: bool = False,
    return_data: bool = False,
) -> ValidationReport | tuple[ValidationReport, object]:
    """Validate input data against the required schema."""
    return _impl().validate_contract(
        data,
        panel_contract,
        apply_aggregation,
        return_data,
    )


__all__ = ["validate_contract", "normalize_panel_columns"]
