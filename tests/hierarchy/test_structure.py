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

    def test_to_s_df_and_tags(self) -> None:
        """Test S_df and tags export ordering."""
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A", "B"],
                "A": ["A1", "A2"],
            },
            bottom_nodes=["A1", "A2", "B"],
            s_matrix=np.array([
                #A1 A2  B
                [1, 1, 0],  # A
                [1, 0, 0],  # A1
                [0, 1, 0],  # A2
                [0, 0, 1],  # B
                [1, 1, 1],  # Total
            ]),
        )

        order = structure.node_order()
        assert order[-3:] == ["A1", "A2", "B"]

        s_df = structure.to_s_df()
        assert list(s_df.columns) == ["unique_id", "A1", "A2", "B"]
        assert list(s_df["unique_id"]) == order

        tags = structure.to_tags()
        assert tags["level_0"].tolist() == ["Total"]
        assert set(tags["level_1"].tolist()) == {"A", "B"}
        assert set(tags["level_2"].tolist()) == {"A1", "A2"}

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

        assert set(structure.bottom_nodes) == {"store__A", "store__B", "store__C", "store__D"}
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

        assert set(structure.bottom_nodes) == {"city__SF", "city__LA", "city__NYC", "city__BUF"}
        assert structure.get_num_levels() == 3

    def test_empty_columns_raises(self) -> None:
        """Test that empty hierarchy_columns raises error."""
        df = pd.DataFrame({"y": [1, 2, 3]})

        with pytest.raises(ValueError, match="hierarchy_columns cannot be empty"):
            HierarchyStructure.from_dataframe(df, hierarchy_columns=[])

    def test_duplicate_values_across_levels(self) -> None:
        """Test that same value at different hierarchy levels doesn't collide.

        This tests the fix for the duplicate node name collision issue where
        values like "North" appearing at both region and state levels would
        incorrectly share the same key in the aggregation graph.
        """
        # Same value "North" appears at both region and state levels
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "state": ["North", "South", "East", "West"],  # "North" also here!
            "city": ["A", "B", "C", "D"],
            "y": [100, 200, 150, 250],
        })

        structure = HierarchyStructure.from_dataframe(
            df, hierarchy_columns=["region", "state", "city"]
        )

        # With level prefixes, these are distinct nodes
        # region__North and state__North should be different nodes
        assert "region__North" in structure.all_nodes
        assert "state__North" in structure.all_nodes

        # Check that region__North has children state__North and state__South
        region_north_children = structure.get_children("region__North")
        assert "state__North" in region_north_children
        assert "state__South" in region_north_children

        # Check that state__North has child city__A
        state_north_children = structure.get_children("state__North")
        assert "city__A" in state_north_children

        # Bottom nodes should be cities
        assert set(structure.bottom_nodes) == {"city__A", "city__B", "city__C", "city__D"}


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

    def test_multi_level_hierarchy_inference(self) -> None:
        """Test that multi-level hierarchy is correctly inferred from S matrix.

        Tests the fix for the incomplete from_summation_matrix implementation
        where intermediate nodes were being skipped.

        Note: This test uses a hierarchy where all parent nodes have multiple
        children (Region_A has 2 stores, Region_B has 2 stores), ensuring
        distinct contribution patterns for proper inference.
        """
        # 3-level hierarchy: Total -> Region -> Store
        # Each region has multiple stores for distinct contribution patterns
        # node_names = ["Region_A", "Region_B", "Store_A1", "Store_A2", "Store_B1", "Store_B2", "Total"]
        # bottom_nodes = ["Store_A1", "Store_A2", "Store_B1", "Store_B2"]
        s_matrix = np.array([
            # Store_A1, Store_A2, Store_B1, Store_B2
            [1, 1, 0, 0],       # Region_A (contains A1, A2)
            [0, 0, 1, 1],       # Region_B (contains B1, B2)
            [1, 0, 0, 0],       # Store_A1
            [0, 1, 0, 0],       # Store_A2
            [0, 0, 1, 0],       # Store_B1
            [0, 0, 0, 1],       # Store_B2
            [1, 1, 1, 1],       # Total
        ])

        structure = HierarchyStructure.from_summation_matrix(
            s_matrix=s_matrix,
            node_names=["Region_A", "Region_B", "Store_A1", "Store_A2", "Store_B1", "Store_B2", "Total"],
            bottom_node_names=["Store_A1", "Store_A2", "Store_B1", "Store_B2"],
        )

        # Check intermediate nodes are preserved, not flattened
        # Total should have Region_A and Region_B as children (not stores directly)
        total_children = set(structure.get_children("Total"))
        assert total_children == {"Region_A", "Region_B"}, \
            f"Total should have regions as children, got {total_children}"

        # Region_A should have Store_A1 and Store_A2 as children
        region_a_children = set(structure.get_children("Region_A"))
        assert region_a_children == {"Store_A1", "Store_A2"}, \
            f"Region_A should have its stores as children, got {region_a_children}"

        # Region_B should have Store_B1 and Store_B2 as children
        region_b_children = set(structure.get_children("Region_B"))
        assert region_b_children == {"Store_B1", "Store_B2"}, \
            f"Region_B should have Store_B1 and Store_B2 as children, got {region_b_children}"

        # Bottom nodes should have no children
        assert structure.get_children("Store_A1") == []
        assert structure.get_children("Store_A2") == []
        assert structure.get_children("Store_B1") == []
        assert structure.get_children("Store_B2") == []

    def test_hierarchy_depth_preserved(self) -> None:
        """Test that hierarchy depth is preserved correctly.

        Uses a 4-level hierarchy where all parent nodes have multiple children,
        ensuring distinct contribution patterns for proper inference.
        """
        # 4-level hierarchy: Total -> Country -> State -> City
        # 8 bottom cities: Country X has 4 (A1-A4), Country Y has 4 (B1-B4)
        # States: XA (A1,A2), XB (A3,A4), YA (B1,B2), YB (B3,B4)
        bottom_nodes = ["City_A1", "City_A2", "City_A3", "City_A4",
                        "City_B1", "City_B2", "City_B3", "City_B4"]

        s_matrix = np.array([
            # A1, A2, A3, A4, B1, B2, B3, B4
            [1, 1, 1, 1, 0, 0, 0, 0],   # Country_X (4 cities)
            [0, 0, 0, 0, 1, 1, 1, 1],   # Country_Y (4 cities)
            [1, 1, 0, 0, 0, 0, 0, 0],   # State_XA (A1, A2)
            [0, 0, 1, 1, 0, 0, 0, 0],   # State_XB (A3, A4)
            [0, 0, 0, 0, 1, 1, 0, 0],   # State_YA (B1, B2)
            [0, 0, 0, 0, 0, 0, 1, 1],   # State_YB (B3, B4)
            [1, 0, 0, 0, 0, 0, 0, 0],   # City_A1
            [0, 1, 0, 0, 0, 0, 0, 0],   # City_A2
            [0, 0, 1, 0, 0, 0, 0, 0],   # City_A3
            [0, 0, 0, 1, 0, 0, 0, 0],   # City_A4
            [0, 0, 0, 0, 1, 0, 0, 0],   # City_B1
            [0, 0, 0, 0, 0, 1, 0, 0],   # City_B2
            [0, 0, 0, 0, 0, 0, 1, 0],   # City_B3
            [0, 0, 0, 0, 0, 0, 0, 1],   # City_B4
            [1, 1, 1, 1, 1, 1, 1, 1],   # Total
        ])

        node_names = ["Country_X", "Country_Y",
                      "State_XA", "State_XB", "State_YA", "State_YB",
                      "City_A1", "City_A2", "City_A3", "City_A4",
                      "City_B1", "City_B2", "City_B3", "City_B4",
                      "Total"]

        structure = HierarchyStructure.from_summation_matrix(
            s_matrix=s_matrix,
            node_names=node_names,
            bottom_node_names=bottom_nodes,
        )

        # Total should have countries as children
        total_children = set(structure.get_children("Total"))
        assert total_children == {"Country_X", "Country_Y"}, \
            f"Total children: {total_children}"

        # Country_X should have State_XA and State_XB as children
        country_x_children = set(structure.get_children("Country_X"))
        assert country_x_children == {"State_XA", "State_XB"}, \
            f"Country_X children: {country_x_children}"

        # Country_Y should have State_YA and State_YB as children
        country_y_children = set(structure.get_children("Country_Y"))
        assert country_y_children == {"State_YA", "State_YB"}, \
            f"Country_Y children: {country_y_children}"

        # State_XA should have City_A1 and City_A2 as children
        state_xa_children = set(structure.get_children("State_XA"))
        assert state_xa_children == {"City_A1", "City_A2"}, \
            f"State_XA children: {state_xa_children}"
