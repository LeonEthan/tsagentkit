"""Hierarchical forecast reconciliation methods.

Implements various reconciliation strategies to ensure coherent forecasts
across the hierarchy.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from .aggregation import (
    create_bottom_up_matrix,
    create_middle_out_matrix,
    create_ols_matrix,
    create_top_down_matrix,
    create_wls_matrix,
)
from .structure import HierarchyStructure


class ReconciliationMethod(Enum):
    """Available reconciliation methods.

    - BOTTOM_UP: Aggregate bottom-level forecasts up the hierarchy
    - TOP_DOWN: Distribute top-level forecasts down using proportions
    - MIDDLE_OUT: Use forecasts at a middle level as pivot
    - OLS: Ordinary Least Squares (structural) reconciliation
    - WLS: Weighted Least Squares reconciliation
    - MIN_TRACE: Minimum Trace (optimal) reconciliation
    """

    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    MIDDLE_OUT = "middle_out"
    OLS = "ols"
    WLS = "wls"
    MIN_TRACE = "min_trace"


class Reconciler:
    """Hierarchical forecast reconciliation engine.

    Implements various reconciliation strategies to ensure
    coherent forecasts across the hierarchy.

    Example:
        >>> reconciler = Reconciler(
        ...     method=ReconciliationMethod.MIN_TRACE,
        ...     structure=hierarchy_structure,
        ... )
        >>> reconciled = reconciler.reconcile(
        ...     base_forecasts=forecasts,
        ...     residuals=residuals,
        ... )

    Attributes:
        method: Reconciliation method to use
        structure: Hierarchy structure
        _projection_matrix: Cached projection matrix
    """

    def __init__(
        self,
        method: ReconciliationMethod,
        structure: HierarchyStructure,
    ):
        """Initialize reconciler.

        Args:
            method: Reconciliation method
            structure: Hierarchy structure
        """
        self.method = method
        self.structure = structure
        self._projection_matrix: np.ndarray | None = None
        self._cached_params: dict = {}

    def reconcile(
        self,
        base_forecasts: np.ndarray,
        fitted_values: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Reconcile base forecasts to be hierarchy-consistent.

        Args:
            base_forecasts: Array of shape (n_nodes, horizon) or (n_nodes,)
            fitted_values: Fitted values for MinT (n_nodes, n_obs)
            residuals: Residuals for MinT variance estimation (n_nodes, n_obs)
            **kwargs: Additional method-specific parameters
                - middle_level: int (for MIDDLE_OUT)
                - proportions: dict (for TOP_DOWN)
                - weights: np.ndarray (for WLS)

        Returns:
            Reconciled forecasts with same shape as base_forecasts
        """
        # Compute or retrieve projection matrix
        cache_key = self._make_cache_key(fitted_values, residuals, **kwargs)

        if (
            self._projection_matrix is None
            or self._cached_params.get("cache_key") != cache_key
        ):
            self._projection_matrix = self._compute_projection_matrix(
                fitted_values, residuals, **kwargs
            )
            self._cached_params = {"cache_key": cache_key}

        # Handle both 1D and 2D inputs
        if base_forecasts.ndim == 1:
            # Single horizon point: (n_nodes,)
            reconciled = self.structure.s_matrix @ self._projection_matrix @ base_forecasts
            return reconciled
        else:
            # Multiple horizon points: (n_nodes, horizon)
            reconciled = self.structure.s_matrix @ self._projection_matrix @ base_forecasts
            return reconciled

    def _make_cache_key(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> str:
        """Create a cache key for the projection matrix."""
        key_parts = [self.method.value]

        if fitted_values is not None:
            key_parts.append(f"fitted_{fitted_values.shape}")
        if residuals is not None:
            key_parts.append(f"resid_{residuals.shape}")

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{v}")

        return "_".join(key_parts)

    def _compute_projection_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """Compute projection matrix for the reconciliation method."""
        method_map: dict[ReconciliationMethod, Callable] = {
            ReconciliationMethod.BOTTOM_UP: self._bottom_up_matrix,
            ReconciliationMethod.TOP_DOWN: self._top_down_matrix,
            ReconciliationMethod.MIDDLE_OUT: self._middle_out_matrix,
            ReconciliationMethod.OLS: self._ols_matrix,
            ReconciliationMethod.WLS: self._wls_matrix,
            ReconciliationMethod.MIN_TRACE: self._mint_matrix,
        }

        func = method_map[self.method]
        return func(fitted_values, residuals, **kwargs)

    def _bottom_up_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """Bottom-up projection matrix."""
        return create_bottom_up_matrix(self.structure)

    def _top_down_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """Top-down projection matrix."""
        proportions = kwargs.get("proportions")
        return create_top_down_matrix(
            self.structure,
            proportions=proportions,
            historical_data=fitted_values,
        )

    def _middle_out_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """Middle-out projection matrix."""
        middle_level = kwargs.get("middle_level", 1)
        return create_middle_out_matrix(self.structure, middle_level=middle_level)

    def _ols_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """OLS (structural) projection matrix."""
        return create_ols_matrix(self.structure)

    def _wls_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """WLS projection matrix."""
        weights = kwargs.get("weights")
        if weights is None:
            # Default to equal weights
            weights = np.ones(len(self.structure.all_nodes))
        return create_wls_matrix(self.structure, weights)

    def _mint_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        """MinT (minimum trace) optimal reconciliation.

        Uses estimated variance-covariance matrix of forecast errors
        to produce optimal reconciled forecasts.

        P_mint = (S' W^(-1) S)^(-1) S' W^(-1)

        where W is the variance-covariance matrix of base forecast errors.
        """
        if residuals is None:
            # Fall back to OLS if no residuals provided
            return self._ols_matrix(fitted_values, residuals, **kwargs)

        # Estimate W from residuals
        w = self._estimate_w(residuals)

        # Add regularization for numerical stability
        w = self._regularize_w(w)

        s = self.structure.s_matrix.astype(float)
        w_inv = np.linalg.inv(w)

        # Solve for P_mint
        s_w_inv_s = s.T @ w_inv @ s

        # Check if invertible
        if np.linalg.matrix_rank(s_w_inv_s) < s_w_inv_s.shape[0]:
            # Use pseudo-inverse if singular
            p_matrix = np.linalg.pinv(s_w_inv_s) @ s.T @ w_inv
        else:
            p_matrix = np.linalg.inv(s_w_inv_s) @ s.T @ w_inv

        return p_matrix

    def _estimate_w(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate variance-covariance matrix from residuals.

        Args:
            residuals: Residuals of shape (n_nodes, n_obs)

        Returns:
            Estimated W matrix of shape (n_nodes, n_nodes)
        """
        # Sample covariance matrix
        if residuals.ndim == 1:
            # Single observation per series
            return np.diag(residuals ** 2)

        return np.cov(residuals)

    def _regularize_w(
        self,
        w: np.ndarray,
        lambda_reg: float = 0.01,
    ) -> np.ndarray:
        """Regularize covariance matrix for numerical stability.

        Args:
            w: Covariance matrix
            lambda_reg: Regularization parameter

        Returns:
            Regularized matrix
        """
        # Add small diagonal term for stability
        return w + lambda_reg * np.eye(w.shape[0]) * np.trace(w) / w.shape[0]


def reconcile_forecasts(
    base_forecasts: pd.DataFrame,
    structure: HierarchyStructure,
    method: ReconciliationMethod,
    fitted_values: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """High-level function for hierarchical reconciliation.

    Args:
        base_forecasts: DataFrame with columns [unique_id, ds, yhat, ...]
        structure: Hierarchy structure
        method: Reconciliation method
        fitted_values: Optional fitted values for MinT
        residuals: Optional residuals for MinT
        **kwargs: Additional method-specific parameters

    Returns:
        Reconciled forecasts DataFrame
    """
    reconciler = Reconciler(method, structure)

    # Convert DataFrame to matrix format
    # Assume base_forecasts is in long format with unique_id, ds, yhat
    pivot_df = base_forecasts.pivot(index="unique_id", columns="ds", values="yhat")

    # Ensure ordering matches structure
    pivot_df = pivot_df.reindex(structure.all_nodes)

    # Convert to numpy
    forecast_matrix = pivot_df.values  # (n_nodes, horizon)

    # Convert fitted_values and residuals if provided
    fitted_matrix = None
    if fitted_values is not None:
        fitted_pivot = fitted_values.pivot(
            index="unique_id", columns="ds", values="yhat"
        )
        fitted_pivot = fitted_pivot.reindex(structure.all_nodes)
        fitted_matrix = fitted_pivot.values

    residual_matrix = None
    if residuals is not None:
        resid_pivot = residuals.pivot(
            index="unique_id", columns="ds", values="residual"
        )
        resid_pivot = resid_pivot.reindex(structure.all_nodes)
        residual_matrix = resid_pivot.values

    # Reconcile
    reconciled_matrix = reconciler.reconcile(
        forecast_matrix,
        fitted_values=fitted_matrix,
        residuals=residual_matrix,
        **kwargs,
    )

    # Convert back to DataFrame
    reconciled_df = pd.DataFrame(
        reconciled_matrix,
        index=pivot_df.index,
        columns=pivot_df.columns,
    ).reset_index()

    # Melt back to long format
    reconciled_df = reconciled_df.melt(
        id_vars="unique_id",
        var_name="ds",
        value_name="yhat",
    )

    # Add reconciliation method as metadata
    reconciled_df["reconciliation_method"] = method.value

    return reconciled_df
