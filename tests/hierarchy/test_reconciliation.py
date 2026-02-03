"""Tests for hierarchical reconciliation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit.hierarchy import HierarchyStructure, Reconciler, ReconciliationMethod


@pytest.fixture
def simple_structure():
    """Create a simple 2-level hierarchy."""
    # all_nodes = ["A", "B", "Total"]
    return HierarchyStructure(
        aggregation_graph={
            "Total": ["A", "B"],
        },
        bottom_nodes=["A", "B"],
        s_matrix=np.array([
            #A  B
            [1, 0],  # A (index 0)
            [0, 1],  # B (index 1)
            [1, 1],  # Total (index 2)
        ]),
    )


@pytest.fixture
def three_level_structure():
    """Create a 3-level hierarchy."""
    # all_nodes = ["A", "A1", "A2", "B", "B1", "B2", "Total"]
    return HierarchyStructure(
        aggregation_graph={
            "Total": ["A", "B"],
            "A": ["A1", "A2"],
            "B": ["B1", "B2"],
        },
        bottom_nodes=["A1", "A2", "B1", "B2"],
        s_matrix=np.array([
            #A1 A2 B1 B2
            [1, 1, 0, 0],  # A (index 0)
            [1, 0, 0, 0],  # A1 (index 1)
            [0, 1, 0, 0],  # A2 (index 2)
            [0, 0, 1, 1],  # B (index 3)
            [0, 0, 1, 0],  # B1 (index 4)
            [0, 0, 0, 1],  # B2 (index 5)
            [1, 1, 1, 1],  # Total (index 6)
        ]),
    )


class TestReconciler:
    """Test Reconciler class."""

    def test_initialization(self, simple_structure):
        """Test reconciler initialization."""
        reconciler = Reconciler(ReconciliationMethod.BOTTOM_UP, simple_structure)

        assert reconciler.method == ReconciliationMethod.BOTTOM_UP
        assert reconciler.structure == simple_structure

    def test_bottom_up_reconciliation(self, simple_structure):
        """Test bottom-up reconciliation."""
        reconciler = Reconciler(ReconciliationMethod.BOTTOM_UP, simple_structure)

        # Incoherent base forecasts
        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 15])  # A=4, B=6, Total=15 (but 4+6=10 != 15)

        reconciled = reconciler.reconcile(y)

        # Should be coherent: Total (index 2) = A (index 0) + B (index 1)
        assert reconciled[2] == reconciled[0] + reconciled[1]
        # Bottom forecasts should be unchanged
        assert reconciled[0] == 4
        assert reconciled[1] == 6
        # Total should be adjusted
        assert reconciled[2] == 10

    def test_ols_reconciliation(self, simple_structure):
        """Test OLS reconciliation."""
        reconciler = Reconciler(ReconciliationMethod.OLS, simple_structure)

        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 15])  # A=4, B=6, Total=15
        reconciled = reconciler.reconcile(y)

        # Should be coherent: Total (index 2) = A (index 0) + B (index 1)
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-10

    def test_multi_horizon_reconciliation(self, simple_structure):
        """Test reconciliation with multiple horizons."""
        reconciler = Reconciler(ReconciliationMethod.BOTTOM_UP, simple_structure)

        # 3 nodes x 5 horizons
        # all_nodes = ["A", "B", "Total"]
        y = np.array([
            [4, 5, 6, 7, 8],       # A
            [6, 7, 8, 9, 10],      # B
            [15, 16, 17, 18, 19],  # Total (wrong, should be A+B)
        ])

        reconciled = reconciler.reconcile(y)

        # Should be coherent at each horizon: Total = A + B
        for h in range(5):
            assert reconciled[2, h] == reconciled[0, h] + reconciled[1, h]

    def test_mint_reconciliation(self, simple_structure):
        """Test MinT reconciliation with residuals."""
        reconciler = Reconciler(ReconciliationMethod.MIN_TRACE, simple_structure)

        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 15])  # A=4, B=6, Total=15
        # Residuals for variance estimation
        # Order: A, B, Total
        residuals = np.array([
            [0.5, -0.5, 0, 1, -1],  # A residuals
            [0.5, -0.5, 0, 1, -1],  # B residuals
            [1, -1, 0, 2, -2],      # Total residuals
        ])

        reconciled = reconciler.reconcile(y, residuals=residuals)

        # Should be coherent: Total (index 2) = A (index 0) + B (index 1)
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-6

    def test_mint_fallback_to_ols(self, simple_structure):
        """Test MinT falls back to OLS without residuals."""
        reconciler = Reconciler(ReconciliationMethod.MIN_TRACE, simple_structure)

        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 15])  # A=4, B=6, Total=15
        reconciled = reconciler.reconcile(y, residuals=None)

        # Should still produce coherent results: Total (index 2) = A (index 0) + B (index 1)
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-6


class TestReconciliationMethods:
    """Test different reconciliation methods."""

    def test_all_methods_produce_coherent_forecasts(self, simple_structure):
        """Test that all methods produce coherent forecasts."""
        methods = [
            ReconciliationMethod.BOTTOM_UP,
            ReconciliationMethod.TOP_DOWN,
            ReconciliationMethod.OLS,
            ReconciliationMethod.WLS,
            ReconciliationMethod.MIN_TRACE,
        ]

        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 15])  # A=4, B=6, Total=15
        historical = np.array([[60], [40], [100]])  # A, B, Total

        for method in methods:
            reconciler = Reconciler(method, simple_structure)

            if method == ReconciliationMethod.TOP_DOWN:
                reconciled = reconciler.reconcile(y, fitted_values=historical)
            elif method == ReconciliationMethod.WLS:
                reconciled = reconciler.reconcile(y, weights=np.ones(3))
            else:
                reconciled = reconciler.reconcile(y)

            # Check coherence: Total (index 2) = A (index 0) + B (index 1)
            assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-5, \
                f"{method.value} did not produce coherent forecasts"


class TestReconcileForecastsFunction:
    """Test high-level reconcile_forecasts function."""

    def test_reconcile_dataframe(self, simple_structure):
        """Test reconciling DataFrame forecasts."""
        from tsagentkit.hierarchy.reconciliation import reconcile_forecasts

        # Create forecast DataFrame
        forecasts = pd.DataFrame({
            "unique_id": ["Total", "A", "B"] * 3,
            "ds": ["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "yhat": [15, 4, 6, 16, 5, 7, 17, 6, 8],
        })

        reconciled = reconcile_forecasts(
            forecasts,
            simple_structure,
            ReconciliationMethod.BOTTOM_UP,
        )

        # Check result structure
        assert "unique_id" in reconciled.columns
        assert "ds" in reconciled.columns
        assert "yhat" in reconciled.columns

        # Check coherence
        for ds in pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]):
            total = reconciled[
                (reconciled["unique_id"] == "Total") & (reconciled["ds"] == ds)
            ]["yhat"].values[0]
            a = reconciled[
                (reconciled["unique_id"] == "A") & (reconciled["ds"] == ds)
            ]["yhat"].values[0]
            b = reconciled[
                (reconciled["unique_id"] == "B") & (reconciled["ds"] == ds)
            ]["yhat"].values[0]

            assert abs(total - (a + b)) < 1e-6

    def test_reconciliation_method_enum(self):
        """Test ReconciliationMethod enum values."""
        assert ReconciliationMethod.BOTTOM_UP.value == "bottom_up"
        assert ReconciliationMethod.TOP_DOWN.value == "top_down"
        assert ReconciliationMethod.MIDDLE_OUT.value == "middle_out"
        assert ReconciliationMethod.OLS.value == "ols"
        assert ReconciliationMethod.WLS.value == "wls"
        assert ReconciliationMethod.MIN_TRACE.value == "min_trace"
