"""Tests for series/dataset.py."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset, build_dataset


class TestTSDatasetCreation:
    """Tests for TSDataset creation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame."""
        return pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })

    @pytest.fixture
    def sample_spec(self) -> TaskSpec:
        """Create a sample TaskSpec."""
        return TaskSpec(h=7, freq="D")

    def test_from_dataframe_valid(self, sample_df: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test creating TSDataset from valid DataFrame."""
        dataset = TSDataset.from_dataframe(sample_df, sample_spec)
        assert dataset.n_series == 2
        assert dataset.n_observations == 4
        assert dataset.freq == "D"

    def test_from_dataframe_sorted(self, sample_df: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test that data is sorted by unique_id, ds."""
        # Shuffle the data
        shuffled = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
        dataset = TSDataset.from_dataframe(shuffled, sample_spec, validate=False)  # Skip validation to avoid sort error
        # Should be sorted
        expected_order = dataset.df.sort_values(["unique_id", "ds"])
        pd.testing.assert_frame_equal(dataset.df.reset_index(drop=True), expected_order.reset_index(drop=True))

    def test_from_dataframe_with_sparsity(self, sample_df: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test that sparsity profile is computed."""
        dataset = TSDataset.from_dataframe(sample_df, sample_spec, compute_sparsity=True)
        assert dataset.sparsity_profile is not None
        assert "A" in dataset.sparsity_profile.series_profiles
        assert "B" in dataset.sparsity_profile.series_profiles

    def test_from_dataframe_without_sparsity(self, sample_df: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test creating without sparsity profile."""
        dataset = TSDataset.from_dataframe(sample_df, sample_spec, compute_sparsity=False)
        assert dataset.sparsity_profile is None

    def test_invalid_dataframe(self, sample_spec: TaskSpec) -> None:
        """Test error with invalid DataFrame."""
        from tsagentkit.contracts import EContractInvalidType

        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": ["not-a-date"],
            "y": [1.0],
        })
        with pytest.raises(EContractInvalidType):
            TSDataset.from_dataframe(df, sample_spec, validate=True)

    def test_missing_column(self, sample_spec: TaskSpec) -> None:
        """Test error with missing column."""
        from tsagentkit.contracts import EContractMissingColumn

        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": ["2024-01-01"],
            # Missing y
        })
        with pytest.raises(EContractMissingColumn):
            TSDataset.from_dataframe(df, sample_spec, validate=True)


class TestTSDatasetProperties:
    """Tests for TSDataset properties."""

    @pytest.fixture
    def dataset(self) -> TSDataset:
        """Create a sample TSDataset."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B", "B"],
            "ds": list(pd.date_range("2024-01-01", periods=3, freq="D")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        spec = TaskSpec(h=7, freq="D")
        return TSDataset.from_dataframe(df, spec)

    def test_n_series(self, dataset: TSDataset) -> None:
        """Test n_series property."""
        assert dataset.n_series == 2

    def test_n_observations(self, dataset: TSDataset) -> None:
        """Test n_observations property."""
        assert dataset.n_observations == 6

    def test_date_range(self, dataset: TSDataset) -> None:
        """Test date_range property."""
        start, end = dataset.date_range
        assert start == pd.Timestamp("2024-01-01")
        assert end == pd.Timestamp("2024-01-03")

    def test_series_ids(self, dataset: TSDataset) -> None:
        """Test series_ids property."""
        assert dataset.series_ids == ["A", "B"]


class TestTSDatasetMethods:
    """Tests for TSDataset methods."""

    @pytest.fixture
    def dataset(self) -> TSDataset:
        """Create a sample TSDataset."""
        # Both series have same dates Jan 1-3
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B", "B"],
            "ds": list(pd.date_range("2024-01-01", periods=3, freq="D")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        spec = TaskSpec(h=7, freq="D")
        return TSDataset.from_dataframe(df, spec)

    def test_get_series(self, dataset: TSDataset) -> None:
        """Test get_series method."""
        series_a = dataset.get_series("A")
        assert len(series_a) == 3  # Jan 1, 2, 3
        assert all(series_a["unique_id"] == "A")

    def test_filter_series(self, dataset: TSDataset) -> None:
        """Test filter_series method."""
        filtered = dataset.filter_series(["A"])
        assert filtered.n_series == 1
        assert "A" in filtered.series_ids
        assert "B" not in filtered.series_ids
        # Original dataset unchanged
        assert dataset.n_series == 2

    def test_filter_dates_by_start(self, dataset: TSDataset) -> None:
        """Test filter_dates with start date."""
        filtered = dataset.filter_dates(start="2024-01-03")
        assert filtered.n_observations == 2  # 1 day * 2 series

    def test_filter_dates_by_end(self, dataset: TSDataset) -> None:
        """Test filter_dates with end date."""
        filtered = dataset.filter_dates(end="2024-01-02")
        assert filtered.n_observations == 4  # 2 days * 2 series

    def test_filter_dates_by_range(self, dataset: TSDataset) -> None:
        """Test filter_dates with both start and end."""
        filtered = dataset.filter_dates(start="2024-01-02", end="2024-01-03")
        assert filtered.n_observations == 4  # 2 days * 2 series

    def test_split_train_test_by_date(self, dataset: TSDataset) -> None:
        """Test split_train_test with date."""
        train, test = dataset.split_train_test(test_start="2024-01-03")
        # Train: Jan 1-2
        assert train.n_observations == 4
        # Test: Jan 3
        assert test.n_observations == 2

    def test_split_train_test_by_size(self, dataset: TSDataset) -> None:
        """Test split_train_test with size."""
        train, test = dataset.split_train_test(test_size=1)
        # Last 1 observation per series for test
        assert test.n_observations == 2
        assert train.n_observations == 4


class TestTSDatasetImmutability:
    """Tests for TSDataset immutability."""

    def test_dataclass_frozen(self) -> None:
        """Test that TSDataset is frozen."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.date_range("2024-01-01", periods=1),
            "y": [1.0],
        })
        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # Attempting to modify should fail
        with pytest.raises(FrozenInstanceError):
            dataset.n_series = 5  # type: ignore

    def test_operations_return_new_instances(self) -> None:
        """Test that operations return new instances."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # filter_series returns new instance
        filtered = dataset.filter_series(["A"])
        assert filtered is not dataset
        assert filtered.df is not dataset.df

        # filter_dates returns new instance
        date_filtered = dataset.filter_dates(end="2024-01-02")
        assert date_filtered is not dataset

        # split returns new instances
        train, test = dataset.split_train_test(test_size=1)
        assert train is not dataset
        assert test is not dataset


class TestBuildDataset:
    """Tests for build_dataset convenience function."""

    def test_build_dataset(self) -> None:
        """Test build_dataset function."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        spec = TaskSpec(h=7, freq="D")

        dataset = build_dataset(df, spec)
        assert isinstance(dataset, TSDataset)
        assert dataset.n_series == 2

    def test_build_dataset_no_validate(self) -> None:
        """Test build_dataset with validation disabled."""
        # Data with wrong types but validation disabled
        df = pd.DataFrame({
            "unique_id": [1, 1, 2, 2],  # Integers instead of strings
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        spec = TaskSpec(h=7, freq="D")

        dataset = build_dataset(df, spec, validate=False)
        assert isinstance(dataset, TSDataset)


class TestTSDatasetHierarchy:
    """Tests for hierarchical dataset aggregation."""

    def test_aggregate_to_level_with_non_sequential_bottom_nodes(self) -> None:
        """Test aggregation when bottom nodes are not first in all_nodes.

        This tests the fix for the S matrix indexing bug where columns
        correspond to bottom_nodes indices, not all_nodes indices.
        """
        from dataclasses import replace

        import numpy as np

        from tsagentkit.hierarchy import HierarchyStructure

        # Create a hierarchy where bottom nodes are NOT the first elements in all_nodes
        # Structure: Total -> A, B; A -> A1, A2
        # all_nodes = ["A", "A1", "A2", "B", "Total"]
        # bottom_nodes = ["A1", "A2", "B"] (indices 1, 2, 3 in all_nodes, but 0, 1, 2 in bottom_nodes)
        structure = HierarchyStructure(
            aggregation_graph={
                "Total": ["A", "B"],
                "A": ["A1", "A2"],
            },
            bottom_nodes=["A1", "A2", "B"],
            s_matrix=np.array([
                # A1  A2   B  (bottom_nodes order)
                [1, 1, 0],  # A (index 0 in all_nodes)
                [1, 0, 0],  # A1 (index 1 in all_nodes, but 0 in bottom_nodes)
                [0, 1, 0],  # A2 (index 2 in all_nodes, but 1 in bottom_nodes)
                [0, 0, 1],  # B (index 3 in all_nodes, but 2 in bottom_nodes)
                [1, 1, 1],  # Total (index 4 in all_nodes)
            ]),
        )

        # Create dataset with bottom-level data
        df = pd.DataFrame({
            "unique_id": ["A1", "A1", "A1", "A2", "A2", "A2", "B", "B", "B"],
            "ds": list(pd.date_range("2024-01-01", periods=3, freq="D")) * 3,
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        })

        spec = TaskSpec(h=3, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)
        dataset = replace(dataset, hierarchy=structure)

        # Aggregate to level 1 (intermediate nodes: A, B)
        aggregated = dataset.aggregate_to_level(1)

        # Verify structure
        assert aggregated.hierarchy is not None
        assert set(aggregated.series_ids) == {"A", "B"}

        # Verify aggregation: A should be A1 + A2 = 50, 70, 90
        a_data = aggregated.df[aggregated.df["unique_id"] == "A"].sort_values("ds")
        assert a_data["y"].tolist() == [50.0, 70.0, 90.0]

        # B should remain as is = 70, 80, 90
        b_data = aggregated.df[aggregated.df["unique_id"] == "B"].sort_values("ds")
        assert b_data["y"].tolist() == [70.0, 80.0, 90.0]

    def test_aggregate_to_level_root(self) -> None:
        """Test aggregation to root level."""
        from dataclasses import replace

        import numpy as np

        from tsagentkit.hierarchy import HierarchyStructure

        # Simple 2-level hierarchy
        structure = HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],  # A
                [0, 1],  # B
                [1, 1],  # Total
            ]),
        )

        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": list(pd.date_range("2024-01-01", periods=2, freq="D")) * 2,
            "y": [10.0, 20.0, 30.0, 40.0],
        })

        spec = TaskSpec(h=2, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)
        dataset = replace(dataset, hierarchy=structure)

        # Aggregate to level 0 (root)
        aggregated = dataset.aggregate_to_level(0)

        assert aggregated.series_ids == ["Total"]
        # Total should be A + B = 40, 60
        total_data = aggregated.df.sort_values("ds")
        assert total_data["y"].tolist() == [40.0, 60.0]

    def test_aggregate_to_level_no_hierarchy_raises(self) -> None:
        """Test that aggregation raises error without hierarchy."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        spec = TaskSpec(h=2, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        with pytest.raises(ValueError, match="Dataset does not have hierarchy"):
            dataset.aggregate_to_level(0)
