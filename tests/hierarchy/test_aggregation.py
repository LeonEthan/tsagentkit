"""Tests for deprecated hierarchy aggregation helpers."""

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


class TestDeprecatedAggregation:
    """Deprecated aggregation helpers should warn and raise."""

    def test_bottom_up_deprecated(self, simple_structure):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(NotImplementedError):
                create_bottom_up_matrix(simple_structure)

    def test_top_down_deprecated(self, simple_structure):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(NotImplementedError):
                create_top_down_matrix(simple_structure, historical_data=np.array([[1]]))

    def test_middle_out_deprecated(self, three_level_structure):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(NotImplementedError):
                create_middle_out_matrix(three_level_structure, middle_level=1)

    def test_ols_deprecated(self, simple_structure):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(NotImplementedError):
                create_ols_matrix(simple_structure)

    def test_wls_deprecated(self, simple_structure):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(NotImplementedError):
                create_wls_matrix(simple_structure, np.ones(3))
