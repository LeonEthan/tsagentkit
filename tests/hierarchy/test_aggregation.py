"""Tests for hierarchy aggregation matrix operations."""

from __future__ import annotations

import numpy as np
import pytest

from tsagentkit.hierarchy import HierarchyStructure
from tsagentkit.hierarchy.aggregation import (
    create_bottom_up_matrix,
    create_middle_out_matrix,
    create_ols_matrix,
    create_top_down_matrix,
    create_wls_matrix,
)


@pytest.fixture
def simple_structure():
    """Create a simple 2-level hierarchy for testing."""
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
    """Create a 3-level hierarchy for testing."""
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


class TestBottomUpMatrix:
    """Test bottom-up projection matrix."""

    def test_shape(self, simple_structure):
        """Test matrix shape."""
        p = create_bottom_up_matrix(simple_structure)
        assert p.shape == (2, 3)  # (n_bottom, n_total)

    def test_extracts_bottom(self, simple_structure):
        """Test that matrix extracts bottom-level forecasts."""
        p = create_bottom_up_matrix(simple_structure)

        # all_nodes = ["A", "B", "Total"]
        # P should select A (index 0) and B (index 1) from [A, B, Total]
        expected = np.array([
            [1, 0, 0],  # Select A
            [0, 1, 0],  # Select B
        ])
        np.testing.assert_array_equal(p, expected)

    def test_reconciliation(self, simple_structure):
        """Test full reconciliation: S @ P @ y."""
        p = create_bottom_up_matrix(simple_structure)
        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 10])  # [A=4, B=6, Total=10]

        # P @ y selects bottom forecasts: [4, 6]
        # S @ [4, 6] = [4, 6, 10] (coherent)
        reconciled = simple_structure.s_matrix @ p @ y

        # Total (index 2) should equal A (index 0) + B (index 1)
        assert reconciled[2] == reconciled[0] + reconciled[1]


class TestTopDownMatrix:
    """Test top-down projection matrix."""

    def test_shape(self, simple_structure):
        """Test matrix shape."""
        historical = np.array([
            [100],  # Total
            [60],   # A
            [40],   # B
        ])
        p = create_top_down_matrix(simple_structure, historical_data=historical)
        assert p.shape == (2, 3)  # (n_bottom, n_total)

    def test_proportions_from_historical(self, simple_structure):
        """Test proportions computed from historical data."""
        # all_nodes = ["A", "B", "Total"]
        historical = np.array([
            [60, 66, 72],     # A (60%)
            [40, 44, 48],     # B (40%)
            [100, 110, 120],  # Total
        ])
        p = create_top_down_matrix(simple_structure, historical_data=historical)

        # Top forecast is distributed according to proportions
        # A gets 60%, B gets 40%
        # p[0, 2] is the proportion for A from Total (index 2)
        # p[1, 2] is the proportion for B from Total (index 2)
        np.testing.assert_almost_equal(p[0, 2], 0.6, decimal=1)
        np.testing.assert_almost_equal(p[1, 2], 0.4, decimal=1)

    def test_explicit_proportions(self, simple_structure):
        """Test with explicit proportions."""
        proportions = {"A": 0.7, "B": 0.3}
        p = create_top_down_matrix(simple_structure, proportions=proportions)

        # Check proportions are used
        # all_nodes = ["A", "B", "Total"], so Total is at index 2
        np.testing.assert_almost_equal(p[0, 2], 0.7, decimal=2)
        np.testing.assert_almost_equal(p[1, 2], 0.3, decimal=2)

    def test_no_data_raises(self, simple_structure):
        """Test that error is raised without proportions or historical data."""
        with pytest.raises(ValueError, match="Must provide either"):
            create_top_down_matrix(simple_structure)


class TestMiddleOutMatrix:
    """Test middle-out projection matrix."""

    def test_shape(self, three_level_structure):
        """Test matrix shape."""
        p = create_middle_out_matrix(three_level_structure, middle_level=1)
        assert p.shape == (4, 7)  # (n_bottom, n_total)

    def test_invalid_level_raises(self, three_level_structure):
        """Test that invalid level raises error."""
        with pytest.raises(ValueError, match="middle_level"):
            create_middle_out_matrix(three_level_structure, middle_level=10)

        with pytest.raises(ValueError, match="middle_level"):
            create_middle_out_matrix(three_level_structure, middle_level=-1)


class TestOLSMatrix:
    """Test OLS projection matrix."""

    def test_shape(self, simple_structure):
        """Test matrix shape."""
        p = create_ols_matrix(simple_structure)
        assert p.shape == (2, 3)  # (n_bottom, n_total)

    def test_reconciliation(self, simple_structure):
        """Test OLS reconciliation produces coherent forecasts."""
        p = create_ols_matrix(simple_structure)
        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 10])  # [A=4, B=6, Total=10]

        reconciled = simple_structure.s_matrix @ p @ y

        # Should be coherent: Total (index 2) = A (index 0) + B (index 1)
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-10


class TestWLSMatrix:
    """Test WLS projection matrix."""

    def test_shape(self, simple_structure):
        """Test matrix shape."""
        weights = np.array([1.0, 1.0, 1.0])
        p = create_wls_matrix(simple_structure, weights)
        assert p.shape == (2, 3)  # (n_bottom, n_total)

    def test_equal_weights_like_ols(self, simple_structure):
        """Test that equal weights produces similar results to OLS."""
        weights = np.ones(3)
        p_wls = create_wls_matrix(simple_structure, weights)
        p_ols = create_ols_matrix(simple_structure)

        y = np.array([10, 4, 6])
        reconciled_wls = simple_structure.s_matrix @ p_wls @ y
        reconciled_ols = simple_structure.s_matrix @ p_ols @ y

        # Should be very close (allowing for numerical differences)
        np.testing.assert_almost_equal(reconciled_wls, reconciled_ols, decimal=5)

    def test_different_weights(self, simple_structure):
        """Test that different weights affect reconciliation."""
        # Give more weight to bottom level
        # all_nodes = ["A", "B", "Total"]
        weights = np.array([1.0, 1.0, 0.1])  # High weight for A, B; low for Total
        p = create_wls_matrix(simple_structure, weights)

        # all_nodes = ["A", "B", "Total"]
        y = np.array([4, 6, 12])  # A=4, B=6, Total=12 (12 != 10)
        reconciled = simple_structure.s_matrix @ p @ y

        # Should be coherent: Total (index 2) = A (index 0) + B (index 1)
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-10
        # With high weights on bottom, they should be close to original
        assert abs(reconciled[0] - 4) < 1.0  # A close to original
        assert abs(reconciled[1] - 6) < 1.0  # B close to original
