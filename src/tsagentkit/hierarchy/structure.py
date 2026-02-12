"""Hierarchy structure definition for hierarchical time series.

Defines the aggregation relationships between time series in a hierarchy
and provides utilities for validation and navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class HierarchyStructure:
    """Defines hierarchical relationships between time series.

    Represents the aggregation structure where bottom-level series
    sum up to higher-level series. The structure is encoded using:
    - aggregation_graph: Parent -> children mapping
    - bottom_nodes: Leaf nodes (no children)
    - s_matrix: Summation matrix (n_total x n_bottom)

    Example structure for retail:
        Total
        ├── Region_North
        │   ├── Store_A
        │   └── Store_B
        └── Region_South
            ├── Store_C
            └── Store_D

    Attributes:
        aggregation_graph: Mapping from parent to list of children
        bottom_nodes: List of bottom-level (leaf) node names
        s_matrix: Summation matrix where S[i,j] = 1 if bottom node j
                 contributes to node i
        all_nodes: List of all nodes (computed automatically)

    Example:
        >>> structure = HierarchyStructure(
        ...     aggregation_graph={
        ...         "Total": ["Region_North", "Region_South"],
        ...         "Region_North": ["Store_A", "Store_B"],
        ...         "Region_South": ["Store_C", "Store_D"],
        ...     },
        ...     bottom_nodes=["Store_A", "Store_B", "Store_C", "Store_D"],
        ...     s_matrix=s_matrix,  # 7x4 matrix
        ... )
    """

    # Mapping from parent to children
    aggregation_graph: dict[str, list[str]]

    # Bottom-level nodes (leaf nodes)
    bottom_nodes: list[str]

    # All nodes in the hierarchy (computed)
    all_nodes: list[str] = field(init=False)

    # Aggregation matrix S (numpy array)
    # Shape: (n_total, n_bottom)
    # where S[i, j] = 1 if bottom node j contributes to node i
    s_matrix: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        """Validate the hierarchy structure after creation."""
        # Check for empty hierarchy first
        if not self.bottom_nodes:
            raise ValueError("bottom_nodes cannot be empty")

        # Compute all nodes first (needed for validation)
        nodes = set(self.bottom_nodes)
        for parent, children in self.aggregation_graph.items():
            nodes.add(parent)
            nodes.update(children)

        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, "all_nodes", sorted(nodes))

        # Validate structure
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate hierarchy structure is consistent.

        Raises:
            ValueError: If structure is invalid
        """
        # Check all children exist (before S matrix validation)
        for parent, children in self.aggregation_graph.items():
            for child in children:
                if child not in self.all_nodes:
                    raise ValueError(
                        f"Child '{child}' of parent '{parent}' not found in hierarchy"
                    )

        # Check S matrix dimensions
        n_total = len(self.all_nodes)
        n_bottom = len(self.bottom_nodes)
        if self.s_matrix.shape != (n_total, n_bottom):
            raise ValueError(
                f"S matrix shape {self.s_matrix.shape} doesn't match "
                f"expected ({n_total}, {n_bottom})"
            )

        # Check bottom nodes have no children
        for node in self.bottom_nodes:
            if node in self.aggregation_graph:
                raise ValueError(
                    f"Bottom node '{node}' cannot have children in aggregation_graph"
                )

        # Check S matrix values are valid (0 or 1)
        if not np.all(np.isin(self.s_matrix, [0, 1])):
            raise ValueError("S matrix must contain only 0s and 1s")

        # Check each bottom node contributes to exactly one bottom position
        for j, bottom_node in enumerate(self.bottom_nodes):
            bottom_idx = self.all_nodes.index(bottom_node)
            if self.s_matrix[bottom_idx, j] != 1:
                raise ValueError(
                    f"Bottom node '{bottom_node}' must have S[{bottom_idx}, {j}] = 1"
                )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        hierarchy_columns: list[str],
        value_column: str = "y",
    ) -> HierarchyStructure:
        """Build hierarchy structure from DataFrame with hierarchical columns.

        Automatically constructs the aggregation graph and summation matrix
        from hierarchical identifiers in the DataFrame.

        Args:
            df: DataFrame with hierarchical identifiers
            hierarchy_columns: Columns defining hierarchy (top to bottom)
            value_column: Column containing values (for validation)

        Returns:
            HierarchyStructure built from the data

        Example:
            >>> df = pd.DataFrame({
            ...     "country": ["US", "US", "US", "US"],
            ...     "state": ["CA", "CA", "NY", "NY"],
            ...     "city": ["SF", "LA", "NYC", "BUF"],
            ...     "y": [100, 200, 300, 50]
            ... })
            >>> structure = HierarchyStructure.from_dataframe(
            ...     df, ["country", "state", "city"]
            ... )
        """
        if not hierarchy_columns:
            raise ValueError("hierarchy_columns cannot be empty")

        # Get unique combinations
        unique_combos = df[hierarchy_columns].drop_duplicates()

        # Build aggregation graph with level-prefixed keys to avoid collisions
        # Example: "country_US", "state_CA" instead of just "US", "CA"
        aggregation_graph: dict[str, list[str]] = {}
        level_values: dict[str, set[str]] = {col: set() for col in hierarchy_columns}

        for level_idx in range(len(hierarchy_columns) - 1):
            parent_col = hierarchy_columns[level_idx]
            child_col = hierarchy_columns[level_idx + 1]

            # Group by immediate parent column to find children
            for parent_value, group in unique_combos.groupby(parent_col):
                # Use level-prefixed key to prevent collisions across levels
                parent_key = f"{parent_col}__{parent_value}"
                # Children also need to be prefixed with their level
                children = [
                    f"{child_col}__{v}" for v in group[child_col].unique().tolist()
                ]

                if parent_key not in aggregation_graph:
                    aggregation_graph[parent_key] = []
                aggregation_graph[parent_key].extend(children)
                aggregation_graph[parent_key] = list(
                    dict.fromkeys(aggregation_graph[parent_key])
                )  # Remove duplicates, preserve order

                # Track values at each level for validation
                level_values[parent_col].add(str(parent_value))
                level_values[child_col].update(group[child_col].unique().tolist())

        # Bottom nodes are the unique values at the lowest level (with level prefix)
        bottom_col = hierarchy_columns[-1]
        bottom_nodes = [
            f"{bottom_col}__{v}" for v in unique_combos[bottom_col].unique().tolist()
        ]

        # Build summation matrix
        all_nodes = _get_all_nodes_from_graph(aggregation_graph, bottom_nodes)
        s_matrix = _build_summation_matrix(
            all_nodes, bottom_nodes, aggregation_graph
        )

        return cls(
            aggregation_graph=aggregation_graph,
            bottom_nodes=bottom_nodes,
            s_matrix=s_matrix,
        )

    @classmethod
    def from_summation_matrix(
        cls,
        s_matrix: np.ndarray,
        node_names: list[str],
        bottom_node_names: list[str],
    ) -> HierarchyStructure:
        """Build hierarchy from explicit summation matrix.

        Infers the aggregation graph by analyzing which nodes' bottom contributions
        are strict subsets of others. This works best when parent nodes have
        distinct contribution patterns from their children (e.g., parents aggregate
        multiple children).

        Args:
            s_matrix: Summation matrix (n_nodes x n_bottom)
            node_names: Names for all nodes (length n_nodes)
            bottom_node_names: Names for bottom nodes (length n_bottom)

        Returns:
            HierarchyStructure

        Note:
            When a parent node has only a single child and they have identical
            bottom contributions, the hierarchy may be flattened. This is a
            limitation of inferring structure from the S matrix alone. For such
            cases, use explicit HierarchyStructure construction.
        """
        if len(node_names) != s_matrix.shape[0]:
            raise ValueError(
                f"node_names length {len(node_names)} doesn't match "
                f"S matrix rows {s_matrix.shape[0]}"
            )
        if len(bottom_node_names) != s_matrix.shape[1]:
            raise ValueError(
                f"bottom_node_names length {len(bottom_node_names)} doesn't match "
                f"S matrix columns {s_matrix.shape[1]}"
            )

        # Infer aggregation graph from S matrix
        # Build proper multi-level structure by detecting intermediate nodes
        aggregation_graph: dict[str, list[str]] = {}

        # Build a mapping from each node's contributions to bottom nodes
        node_to_bottoms: dict[str, set[int]] = {}
        for i, node in enumerate(node_names):
            # Get indices of bottom nodes that contribute to this node
            contributing = set(j for j in range(len(bottom_node_names)) if s_matrix[i, j] == 1)
            node_to_bottoms[node] = contributing

        bottom_set = set(bottom_node_names)
        non_bottom_nodes = [n for n in node_names if n not in bottom_set]

        # For each non-bottom node, find its direct children
        for parent_node in non_bottom_nodes:
            parent_bottoms = node_to_bottoms[parent_node]

            children = []
            for child_node in node_names:
                if child_node == parent_node:
                    continue

                child_bottoms = node_to_bottoms[child_node]
                if not child_bottoms:
                    continue

                # A child must have contributions strictly contained in parent's
                if not child_bottoms <= parent_bottoms:
                    continue  # Not a descendant at all

                if child_bottoms == parent_bottoms:
                    continue  # Same contributions, not a child

                # Check this is a DIRECT child (no intermediate exists)
                is_direct_child = True
                for intermediate in node_names:
                    if intermediate in (parent_node, child_node):
                        continue

                    intermediate_bottoms = node_to_bottoms[intermediate]
                    if not intermediate_bottoms:
                        continue

                    # If intermediate is strictly between parent and child
                    if child_bottoms < intermediate_bottoms < parent_bottoms:
                        is_direct_child = False
                        break

                if is_direct_child:
                    children.append(child_node)

            if children:
                aggregation_graph[parent_node] = children

        # Reorder S matrix rows to match sorted all_nodes order expected by validation
        # First create the structure to determine all_nodes order
        temp_graph = aggregation_graph.copy()
        all_nodes_set = set(bottom_node_names)
        for parent, children in temp_graph.items():
            all_nodes_set.add(parent)
            all_nodes_set.update(children)
        all_nodes_sorted = sorted(all_nodes_set)

        # Build row permutation: new_row[i] = old_row[perm[i]]
        name_to_old_row = {name: i for i, name in enumerate(node_names)}
        perm = [name_to_old_row[name] for name in all_nodes_sorted]
        s_matrix_reordered = s_matrix[perm, :]

        return cls(
            aggregation_graph=aggregation_graph,
            bottom_nodes=bottom_node_names,
            s_matrix=s_matrix_reordered,
        )

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes of a given node.

        Args:
            node: Node name

        Returns:
            List of parent node names

        Raises:
            ValueError: If node is not in hierarchy
        """
        if node not in self.all_nodes:
            raise ValueError(f"Node '{node}' not in hierarchy")

        parents = []
        for parent, children in self.aggregation_graph.items():
            if node in children:
                parents.append(parent)
        return parents

    def get_children(self, node: str) -> list[str]:
        """Get child nodes of a given node.

        Args:
            node: Node name

        Returns:
            List of child node names
        """
        return self.aggregation_graph.get(node, [])

    def get_level(self, node: str) -> int:
        """Get hierarchy level (0 = root, increasing downward).

        Args:
            node: Node name

        Returns:
            Hierarchy level (0 for root nodes)

        Raises:
            ValueError: If node is not in hierarchy
        """
        if node not in self.all_nodes:
            raise ValueError(f"Node '{node}' not in hierarchy")

        level = 0
        current = node
        parents = self.get_parents(current)

        # Traverse up to find depth
        while parents:
            level += 1
            current = parents[0]  # Use first parent (works for tree structures)
            parents = self.get_parents(current)

        return level

    def get_nodes_at_level(self, level: int) -> list[str]:
        """Get all nodes at a specific hierarchy level.

        Args:
            level: Hierarchy level (0 = root)

        Returns:
            List of node names at that level
        """
        return [node for node in self.all_nodes if self.get_level(node) == level]

    def is_leaf(self, node: str) -> bool:
        """Check if node is a leaf (bottom-level) node.

        Args:
            node: Node name

        Returns:
            True if node is a bottom node
        """
        return node in self.bottom_nodes

    def get_num_levels(self) -> int:
        """Get the number of levels in the hierarchy.

        Returns:
            Maximum level + 1 (since levels start at 0)
        """
        if not self.all_nodes:
            return 0
        return max(self.get_level(node) for node in self.all_nodes) + 1

    def node_order(self) -> list[str]:
        """Return nodes ordered with aggregates first and bottom nodes last."""
        order: list[str] = []
        for level in range(self.get_num_levels()):
            order.extend(self.get_nodes_at_level(level))
        bottom = [n for n in self.bottom_nodes if n in order]
        order = [n for n in order if n not in bottom] + bottom
        return order

    def to_s_df(self, id_col: str = "unique_id") -> pd.DataFrame:
        """Return summation matrix as S_df (rows ordered aggregates -> bottom)."""
        s_df = pd.DataFrame(
            self.s_matrix,
            index=self.all_nodes,
            columns=self.bottom_nodes,
        )
        ordered = self.node_order()
        s_df = s_df.reindex(index=ordered)
        s_df = s_df.reset_index().rename(columns={"index": id_col})
        return s_df

    def to_tags(self, level_prefix: str = "level_") -> dict[str, np.ndarray]:
        """Return hierarchical tags mapping level -> node names."""
        tags: dict[str, np.ndarray] = {}
        order = self.node_order()
        for level in range(self.get_num_levels()):
            level_nodes = [n for n in order if self.get_level(n) == level]
            if level_nodes:
                tags[f"{level_prefix}{level}"] = np.array(level_nodes, dtype=object)
        return tags


def _get_all_nodes_from_graph(
    aggregation_graph: dict[str, list[str]],
    bottom_nodes: list[str],
) -> list[str]:
    """Get all nodes from aggregation graph and bottom nodes."""
    nodes = set(bottom_nodes)
    for parent, children in aggregation_graph.items():
        nodes.add(parent)
        nodes.update(children)
    return sorted(nodes)


def _build_summation_matrix(
    all_nodes: list[str],
    bottom_nodes: list[str],
    aggregation_graph: dict[str, list[str]],
) -> np.ndarray:
    """Build summation matrix from hierarchy definition.

    The S matrix encodes which bottom nodes contribute to each node.
    S[i, j] = 1 if bottom node j contributes to node i.
    """
    n_total = len(all_nodes)
    n_bottom = len(bottom_nodes)

    s_matrix = np.zeros((n_total, n_bottom), dtype=int)

    # Map node names to indices
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    bottom_to_idx = {node: j for j, node in enumerate(bottom_nodes)}

    # For each bottom node, determine all ancestors
    for bottom_node in bottom_nodes:
        bottom_idx = node_to_idx[bottom_node]
        j = bottom_to_idx[bottom_node]

        # Bottom node contributes to itself
        s_matrix[bottom_idx, j] = 1

        # Trace up the hierarchy
        _add_ancestor_contributions(
            bottom_node,
            j,
            s_matrix,
            node_to_idx,
            aggregation_graph,
        )

    return s_matrix


def _add_ancestor_contributions(
    node: str,
    bottom_idx: int,
    s_matrix: np.ndarray,
    node_to_idx: dict[str, int],
    aggregation_graph: dict[str, list[str]],
) -> None:
    """Recursively add contributions for all ancestors of a node."""
    # Find all parents of this node
    parents = [
        parent
        for parent, children in aggregation_graph.items()
        if node in children
    ]

    for parent in parents:
        parent_idx = node_to_idx[parent]
        s_matrix[parent_idx, bottom_idx] = 1
        # Recurse up the hierarchy
        _add_ancestor_contributions(
            parent,
            bottom_idx,
            s_matrix,
            node_to_idx,
            aggregation_graph,
        )
