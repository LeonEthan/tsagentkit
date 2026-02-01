"""Aggregation matrix operations for hierarchical reconciliation.

Provides functions to create projection matrices for different
reconciliation strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .structure import HierarchyStructure


def create_bottom_up_matrix(structure: "HierarchyStructure") -> np.ndarray:
    """Create projection matrix for bottom-up reconciliation.

    Bottom-up takes bottom-level forecasts and aggregates them
    using the summation matrix S to get forecasts for all levels.

    The projection matrix P selects bottom-level forecasts from
    the full forecast vector.

    Returns:
        Projection matrix P of shape (n_bottom, n_total)
        such that ŷ_reconciled = S @ P @ ŷ_base

    Example:
        >>> P = create_bottom_up_matrix(structure)
        >>> reconciled = structure.s_matrix @ P @ base_forecasts
    """
    n_bottom = len(structure.bottom_nodes)
    n_total = len(structure.all_nodes)

    # P extracts bottom-level forecasts
    # Each row selects one bottom node from the full forecast
    p_matrix = np.zeros((n_bottom, n_total))

    for i, node in enumerate(structure.bottom_nodes):
        j = structure.all_nodes.index(node)
        p_matrix[i, j] = 1.0

    return p_matrix


def create_top_down_matrix(
    structure: "HierarchyStructure",
    proportions: dict[str, float] | None = None,
    historical_data: np.ndarray | None = None,
) -> np.ndarray:
    """Create projection matrix for top-down reconciliation.

    Distributes top-level (root) forecasts down the hierarchy using
    historical proportions or provided proportions.

    Args:
        structure: Hierarchy structure
        proportions: Optional dict mapping bottom node names to proportions
        historical_data: Optional historical data to compute proportions from
                         Shape: (n_total, n_timepoints)

    Returns:
        Projection matrix P of shape (n_bottom, n_total)

    Raises:
        ValueError: If neither proportions nor historical_data is provided
    """
    n_bottom = len(structure.bottom_nodes)
    n_total = len(structure.all_nodes)

    # Determine proportions
    if proportions is not None:
        prop_values = np.array([
            proportions.get(node, 0.0) for node in structure.bottom_nodes
        ])
    elif historical_data is not None:
        # Compute proportions from historical data
        # Sum over time for each series
        totals = historical_data.sum(axis=1)

        # Find root node (level 0)
        root_nodes = structure.get_nodes_at_level(0)
        if not root_nodes:
            raise ValueError("No root nodes found in hierarchy")

        root_idx = structure.all_nodes.index(root_nodes[0])
        root_total = totals[root_idx]

        if root_total == 0:
            # Equal proportions if root is zero
            prop_values = np.ones(n_bottom) / n_bottom
        else:
            prop_values = np.array([
                totals[structure.all_nodes.index(node)] / root_total
                for node in structure.bottom_nodes
            ])
    else:
        raise ValueError(
            "Must provide either proportions or historical_data"
        )

    # Normalize proportions
    prop_sum = prop_values.sum()
    if prop_sum > 0:
        prop_values = prop_values / prop_sum

    # Build projection matrix
    # Top-down: only uses the root forecast
    p_matrix = np.zeros((n_bottom, n_total))

    root_nodes = structure.get_nodes_at_level(0)
    if root_nodes:
        root_idx = structure.all_nodes.index(root_nodes[0])
        for i in range(n_bottom):
            p_matrix[i, root_idx] = prop_values[i]

    return p_matrix


def create_middle_out_matrix(
    structure: "HierarchyStructure",
    middle_level: int,
) -> np.ndarray:
    """Create projection matrix for middle-out reconciliation.

    Uses bottom-up from middle level downward, and top-down from
    middle level upward. Forecasts at the middle level are used directly.

    Args:
        structure: Hierarchy structure
        middle_level: The level to use as the pivot (0 = root)

    Returns:
        Projection matrix P of shape (n_bottom, n_total)

    Raises:
        ValueError: If middle_level is invalid
    """
    n_bottom = len(structure.bottom_nodes)
    n_total = len(structure.all_nodes)

    if middle_level < 0 or middle_level >= structure.get_num_levels():
        raise ValueError(
            f"middle_level {middle_level} is out of range "
            f"[0, {structure.get_num_levels()})"
        )

    # Get nodes at middle level
    middle_nodes = structure.get_nodes_at_level(middle_level)

    # Build projection matrix
    p_matrix = np.zeros((n_bottom, n_total))

    for bottom_idx, bottom_node in enumerate(structure.bottom_nodes):
        # Find the ancestor at middle level
        middle_ancestor = _find_ancestor_at_level(
            structure, bottom_node, middle_level
        )

        if middle_ancestor:
            middle_idx = structure.all_nodes.index(middle_ancestor)

            # Compute proportion: this bottom node's share of middle ancestor
            # This is P(bottom | middle) based on historical proportions
            # For simplicity, we use 1.0 here (equal distribution)
            # In practice, this should be computed from historical data
            p_matrix[bottom_idx, middle_idx] = 1.0
        else:
            # Fallback: use bottom node directly
            node_idx = structure.all_nodes.index(bottom_node)
            p_matrix[bottom_idx, node_idx] = 1.0

    return p_matrix


def _find_ancestor_at_level(
    structure: "HierarchyStructure",
    node: str,
    target_level: int,
) -> str | None:
    """Find ancestor of a node at a specific hierarchy level.

    Args:
        structure: Hierarchy structure
        node: Starting node
        target_level: Target hierarchy level

    Returns:
        Ancestor node name, or None if not found
    """
    current_level = structure.get_level(node)

    if current_level == target_level:
        return node

    if current_level < target_level:
        # Node is above target level
        return None

    # Traverse up to find ancestor at target level
    current = node
    while current_level > target_level:
        parents = structure.get_parents(current)
        if not parents:
            return None
        current = parents[0]  # Use first parent
        current_level -= 1

    return current


def create_ols_matrix(structure: "HierarchyStructure") -> np.ndarray:
    """Create projection matrix for OLS (structural) reconciliation.

    OLS reconciliation minimizes the sum of squared errors subject to
    the structural constraints.

    P_ols = (S' S)^(-1) S'

    Args:
        structure: Hierarchy structure

    Returns:
        Projection matrix P of shape (n_bottom, n_total)
    """
    s = structure.s_matrix.astype(float)

    # Compute (S' S)
    s_ts = s.T @ s

    # Check if invertible
    if np.linalg.matrix_rank(s_ts) < s_ts.shape[0]:
        # Use pseudo-inverse if singular
        p_matrix = np.linalg.pinv(s_ts) @ s.T
    else:
        p_matrix = np.linalg.inv(s_ts) @ s.T

    return p_matrix


def create_wls_matrix(
    structure: "HierarchyStructure",
    weights: np.ndarray,
) -> np.ndarray:
    """Create projection matrix for WLS (weighted least squares) reconciliation.

    WLS allows different weights for different levels of the hierarchy.

    P_wls = (S' W^(-1) S)^(-1) S' W^(-1)

    Args:
        structure: Hierarchy structure
        weights: Weight matrix (diagonal) of shape (n_total,)

    Returns:
        Projection matrix P of shape (n_bottom, n_total)
    """
    s = structure.s_matrix.astype(float)
    w_inv = np.diag(1.0 / (weights + 1e-10))  # Add small epsilon for stability

    # Compute (S' W^(-1) S)
    s_w_inv_s = s.T @ w_inv @ s

    # Check if invertible
    if np.linalg.matrix_rank(s_w_inv_s) < s_w_inv_s.shape[0]:
        p_matrix = np.linalg.pinv(s_w_inv_s) @ s.T @ w_inv
    else:
        p_matrix = np.linalg.inv(s_w_inv_s) @ s.T @ w_inv

    return p_matrix
