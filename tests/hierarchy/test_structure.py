"""Tests for hierarchy structure definition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit.hierarchy import HierarchyStructure


class TestHierarchyStructure:
    """Test HierarchyStructure creation and validation."""

    def test_basic_creation(self) -> None:
        """Test basic hierarchy structure creation."""
        structure = HierarchyStructure(
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

        assert structure.all_nodes == ["A", "A1", "A2", "B", "B1", "B2", "Total"]
        assert structure.bottom_nodes == ["A1", "A2", "B1", "B2"]
        assert structure.s_matrix.shape == (7, 4)

    def test_validation_bottom_has_children(self) -> None:
        """Test validation catches bottom nodes with children."""
        # all_nodes = ["A", "A1", "Total"] (3 nodes)
        # bottom_nodes = ["A", "A1"] (2 nodes, but A has children!)
        # S matrix should be 3x2
        with pytest.raises(ValueError, match="Bottom node .* cannot have children"):
            HierarchyStructure(
                aggregation_graph={
                    "Total": ["A"],
                    "A": ["A1"],  # A is listed as bottom but has children
                },
                bottom_nodes=["A", "A1"],  # A should not be bottom
                s_matrix=np.array([
                    # A(bottom)  A1(bottom)
                    [1, 0],  # A (index 0)
                    [0, 1],  # A1 (index 1)
                    [1, 1],  # Total (index 2)
                ]),
            )

    def test_validation_s_matrix_shape(self) -> None:
        """Test validation catches S matrix shape mismatch."""
        with pytest.raises(ValueError, match="S matrix shape"):
            HierarchyStructure(
                aggregation_graph={"Total": ["A"]},
                bottom_nodes=["A"],
                s_matrix=np.array([[1, 0], [0, 1]]),  # Wrong shape
            )

    def test_validation_s_matrix_values(self) -> None:
        """Test validation catches invalid S matrix values."""
        with pytest.raises(ValueError, match="S matrix must contain only 0s and 1s"):
            HierarchyStructure(
                aggregation_graph={"Total": ["A"]},
                bottom_nodes=["A"],
                s_matrix=np.array([[2], [1]]),  # 2 is invalid
            )

    def test_get_parents(self) -> None:
        """Test getting parent nodes."""
        # all_nodes = ["A", "A1", "A2", "B", "Total"]
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A", "B"],
                "A": ["A1", "A2"],
            },
            bottom_nodes=["A1", "A2", "B"],
            s_matrix=np.array([
                #A1 A2  B
                [1, 1, 0],  # A (index 0)
                [1, 0, 0],  # A1 (index 1)
                [0, 1, 0],  # A2 (index 2)
                [0, 0, 1],  # B (index 3)
                [1, 1, 1],  # Total (index 4)
            ]),
        )

        # get_parents returns only direct parents
        assert structure.get_parents("A1") == ["A"]
        assert structure.get_parents("A") == ["Total"]
        assert structure.get_parents("Total") == []

    def test_get_parents_invalid_node(self) -> None:
        """Test getting parents of invalid node raises error."""
        structure = HierarchyStructure(
            aggregation_graph={"Total": ["A"]},
            bottom_nodes=["A"],
            s_matrix=np.array([[1], [1]]),
        )

        with pytest.raises(ValueError, match="Node 'Invalid' not in hierarchy"):
            structure.get_parents("Invalid")

    def test_get_children(self) -> None:
        """Test getting child nodes."""
        # all_nodes = ["A", "A1", "B", "Total"]
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A", "B"],
                "A": ["A1"],
            },
            bottom_nodes=["A1", "B"],
            s_matrix=np.array([
                #A1  B
                [1, 0],  # A (index 0)
                [1, 0],  # A1 (index 1)
                [0, 1],  # B (index 2)
                [1, 1],  # Total (index 3)
            ]),
        )

        assert structure.get_children("Total") == ["A", "B"]
        assert structure.get_children("A") == ["A1"]
        assert structure.get_children("A1") == []

    def test_get_level(self) -> None:
        """Test getting hierarchy level."""
        # all_nodes = ["A", "A1", "B", "Total"]
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A", "B"],
                "A": ["A1"],
            },
            bottom_nodes=["A1", "B"],
            s_matrix=np.array([
                #A1  B
                [1, 0],  # A (index 0)
                [1, 0],  # A1 (index 1)
                [0, 1],  # B (index 2)
                [1, 1],  # Total (index 3)
            ]),
        )

        assert structure.get_level("Total") == 0
        assert structure.get_level("A") == 1
        assert structure.get_level("B") == 1
        assert structure.get_level("A1") == 2

    def test_get_level_invalid_node(self) -> None:
        """Test getting level of invalid node raises error."""
        structure = HierarchyStructure(
            aggregation_graph={"Total": ["A"]},
            bottom_nodes=["A"],
            s_matrix=np.array([[1], [1]]),
        )

        with pytest.raises(ValueError, match="Node 'Invalid' not in hierarchy"):
            structure.get_level("Invalid")

    def test_get_nodes_at_level(self) -> None:
        """Test getting nodes at specific level."""
        # all_nodes = ["A", "A1", "B", "Total"]
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A", "B"],
                "A": ["A1"],
            },
            bottom_nodes=["A1", "B"],
            s_matrix=np.array([
                #A1  B
                [1, 0],  # A (index 0)
                [1, 0],  # A1 (index 1)
                [0, 1],  # B (index 2)
                [1, 1],  # Total (index 3)
            ]),
        )

        assert structure.get_nodes_at_level(0) == ["Total"]
        assert set(structure.get_nodes_at_level(1)) == {"A", "B"}
        assert structure.get_nodes_at_level(2) == ["A1"]
        assert structure.get_nodes_at_level(5) == []  # Beyond max level

    def test_is_leaf(self) -> None:
        """Test checking if node is a leaf."""
        structure = HierarchyStructure(
            aggregation_graph={"Total": ["A"]},
            bottom_nodes=["A"],
            s_matrix=np.array([[1], [1]]),
        )

        assert structure.is_leaf("A") is True
        assert structure.is_leaf("Total") is False

    def test_get_num_levels(self) -> None:
        """Test getting number of hierarchy levels."""
        # all_nodes = ["A", "A1", "Total"]
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A"],
                "A": ["A1"],
            },
            bottom_nodes=["A1"],
            s_matrix=np.array([
                [1],  # A (index 0)
                [1],  # A1 (index 1)
                [1],  # Total (index 2)
            ]),
        )

        assert structure.get_num_levels() == 3

    def test_empty_hierarchy(self) -> None:
        """Test empty hierarchy returns 0 levels."""
        # Note: This would fail validation, but testing edge case
        with pytest.raises(ValueError):
            HierarchyStructure(
                aggregation_graph={},
                bottom_nodes=[],
                s_matrix=np.array([]).reshape(0, 0),
            )


class TestFromDataFrame:
    """Test building hierarchy from DataFrame."""

    def test_simple_hierarchy(self) -> None:
        """Test building hierarchy from simple DataFrame."""
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "store": ["A", "B", "C", "D"],
            "y": [100, 200, 150, 250],
        })

        structure = HierarchyStructure.from_dataframe(
            df, hierarchy_columns=["region", "store"]
        )

        assert set(structure.bottom_nodes) == {"A", "B", "C", "D"}
        assert structure.s_matrix.shape[0] == 6  # Total + 2 regions + 4 stores
        assert structure.s_matrix.shape[1] == 4  # 4 bottom nodes

    def test_three_level_hierarchy(self) -> None:
        """Test building three-level hierarchy."""
        df = pd.DataFrame({
            "country": ["US", "US", "US", "US"],
            "state": ["CA", "CA", "NY", "NY"],
            "city": ["SF", "LA", "NYC", "BUF"],
            "y": [100, 200, 300, 50],
        })

        structure = HierarchyStructure.from_dataframe(
            df, hierarchy_columns=["country", "state", "city"]
        )

        assert set(structure.bottom_nodes) == {"SF", "LA", "NYC", "BUF"}
        assert structure.get_num_levels() == 3

    def test_empty_columns_raises(self) -> None:
        """Test that empty hierarchy_columns raises error."""
        df = pd.DataFrame({"y": [1, 2, 3]})

        with pytest.raises(ValueError, match="hierarchy_columns cannot be empty"):
            HierarchyStructure.from_dataframe(df, hierarchy_columns=[])


class TestFromSummationMatrix:
    """Test building hierarchy from summation matrix."""

    def test_simple_matrix(self) -> None:
        """Test building from simple summation matrix."""
        # all_nodes will be ["A", "B", "Total"] (sorted)
        # S matrix rows must match this order
        s_matrix = np.array([
            [1, 0],  # A
            [0, 1],  # B
            [1, 1],  # Total
        ])

        structure = HierarchyStructure.from_summation_matrix(
            s_matrix=s_matrix,
            node_names=["A", "B", "Total"],
            bottom_node_names=["A", "B"],
        )

        assert structure.all_nodes == ["A", "B", "Total"]
        assert structure.bottom_nodes == ["A", "B"]

    def test_shape_mismatch_raises(self) -> None:
        """Test that shape mismatch raises error."""
        with pytest.raises(ValueError, match="node_names length"):
            HierarchyStructure.from_summation_matrix(
                s_matrix=np.array([[1, 1], [1, 0]]),
                node_names=["Total", "A", "B"],  # 3 names, 2 rows
                bottom_node_names=["A", "B"],
            )

        with pytest.raises(ValueError, match="bottom_node_names length"):
            HierarchyStructure.from_summation_matrix(
                s_matrix=np.array([[1, 1], [1, 0]]),
                node_names=["Total", "A"],
                bottom_node_names=["A", "B", "C"],  # 3 names, 2 cols
            )
