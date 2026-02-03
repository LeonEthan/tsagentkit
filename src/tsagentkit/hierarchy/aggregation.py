"""Deprecated aggregation helpers.

This module previously contained projection-matrix implementations for
hierarchical reconciliation. Those algorithms are now delegated to
`hierarchicalforecast` to keep this package as a thin adapter.
"""

from __future__ import annotations

import warnings

import numpy as np

from .structure import HierarchyStructure

_DEPRECATION_MESSAGE = (
    "tsagentkit.hierarchy.aggregation is deprecated. "
    "Use hierarchicalforecast methods directly instead."
)


def _deprecated(name: str) -> None:
    warnings.warn(
        f"{name} is deprecated. {_DEPRECATION_MESSAGE}",
        DeprecationWarning,
        stacklevel=2,
    )


def create_bottom_up_matrix(structure: HierarchyStructure) -> np.ndarray:  # pragma: no cover
    _deprecated("create_bottom_up_matrix")
    raise NotImplementedError(_DEPRECATION_MESSAGE)


def create_top_down_matrix(  # pragma: no cover
    structure: HierarchyStructure,
    proportions: dict[str, float] | None = None,
    historical_data: np.ndarray | None = None,
) -> np.ndarray:
    _deprecated("create_top_down_matrix")
    raise NotImplementedError(_DEPRECATION_MESSAGE)


def create_middle_out_matrix(  # pragma: no cover
    structure: HierarchyStructure,
    middle_level: int,
) -> np.ndarray:
    _deprecated("create_middle_out_matrix")
    raise NotImplementedError(_DEPRECATION_MESSAGE)


def create_ols_matrix(structure: HierarchyStructure) -> np.ndarray:  # pragma: no cover
    _deprecated("create_ols_matrix")
    raise NotImplementedError(_DEPRECATION_MESSAGE)


def create_wls_matrix(  # pragma: no cover
    structure: HierarchyStructure,
    weights: np.ndarray,
) -> np.ndarray:
    _deprecated("create_wls_matrix")
    raise NotImplementedError(_DEPRECATION_MESSAGE)
