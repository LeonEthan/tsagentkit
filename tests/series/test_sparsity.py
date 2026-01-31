"""Tests for series/sparsity.py."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.series import SparsityClass, compute_sparsity_profile


class TestSparsityClass:
    """Tests for SparsityClass enum."""

    def test_enum_values(self) -> None:
        """Test enum values are strings."""
        assert SparsityClass.REGULAR.value == "regular"
        assert SparsityClass.INTERMITTENT.value == "intermittent"
        assert SparsityClass.SPARSE.value == "sparse"
        assert SparsityClass.COLD_START.value == "cold_start"


class TestComputeSparsityProfile:
    """Tests for compute_sparsity_profile function."""

    def test_regular_series(self) -> None:
        """Test regular series classification."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": np.random.rand(20) + 1,  # No zeros
        })
        profile = compute_sparsity_profile(df)
        assert profile.get_classification("A") == SparsityClass.REGULAR

    def test_intermittent_series(self) -> None:
        """Test intermittent series classification (many zeros)."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [0] * 10 + [1] * 10,  # 50% zeros
        })
        profile = compute_sparsity_profile(df)
        assert profile.get_classification("A") == SparsityClass.INTERMITTENT

    def test_cold_start_series(self) -> None:
        """Test cold start classification (few observations)."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        profile = compute_sparsity_profile(df, min_observations=10)
        assert profile.get_classification("A") == SparsityClass.COLD_START

    def test_sparse_series(self) -> None:
        """Test sparse series classification (gaps in data)."""
        # Create series with gaps
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D")) + \
                list(pd.date_range("2024-01-20", periods=5, freq="D"))
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": dates,
            "y": [1.0] * 10,
        })
        profile = compute_sparsity_profile(df, gap_threshold=0.1)
        assert profile.get_classification("A") == SparsityClass.SPARSE

    def test_multiple_series(self) -> None:
        """Test profile with multiple series."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 20 + ["B"] * 5,
            "ds": list(pd.date_range("2024-01-01", periods=20, freq="D")) + \
                  list(pd.date_range("2024-02-01", periods=5, freq="D")),
            "y": list(np.random.rand(20) + 1) + [0] * 5,
        })
        profile = compute_sparsity_profile(df, min_observations=10)
        assert profile.get_classification("A") == SparsityClass.REGULAR
        assert profile.get_classification("B") == SparsityClass.COLD_START

    def test_get_series_by_class(self) -> None:
        """Test filtering series by classification."""
        df = pd.DataFrame({
            "unique_id": ["regular"] * 20 + ["cold"] * 5,
            "ds": list(pd.date_range("2024-01-01", periods=20, freq="D")) + \
                  list(pd.date_range("2024-02-01", periods=5, freq="D")),
            "y": [1.0] * 25,
        })
        profile = compute_sparsity_profile(df, min_observations=10)
        cold_starts = profile.get_series_by_class(SparsityClass.COLD_START)
        assert "cold" in cold_starts
        assert "regular" not in cold_starts

    def test_has_intermittent(self) -> None:
        """Test has_intermittent method."""
        df_with = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [0] * 10 + [1] * 10,
        })
        profile_with = compute_sparsity_profile(df_with)
        assert profile_with.has_intermittent() is True

        df_without = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [1.0] * 20,
        })
        profile_without = compute_sparsity_profile(df_without)
        assert profile_without.has_intermittent() is False

    def test_has_cold_start(self) -> None:
        """Test has_cold_start method."""
        df_with = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": [1.0] * 5,
        })
        profile_with = compute_sparsity_profile(df_with, min_observations=10)
        assert profile_with.has_cold_start() is True

        df_without = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [1.0] * 20,
        })
        profile_without = compute_sparsity_profile(df_without, min_observations=10)
        assert profile_without.has_cold_start() is False

    def test_dataset_metrics(self) -> None:
        """Test dataset-level metrics."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 20 + ["B"] * 5,
            "ds": list(pd.date_range("2024-01-01", periods=20, freq="D")) + \
                  list(pd.date_range("2024-02-01", periods=5, freq="D")),
            "y": [1.0] * 25,
        })
        profile = compute_sparsity_profile(df, min_observations=10)
        assert profile.dataset_metrics["total_series"] == 2
        assert "classification_counts" in profile.dataset_metrics
        assert "avg_observations" in profile.dataset_metrics


class TestSparsityProfileMetrics:
    """Tests for sparsity metrics computation."""

    def test_zero_ratio(self) -> None:
        """Test zero ratio metric."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": [0] * 3 + [1] * 7,  # 30% zeros
        })
        profile = compute_sparsity_profile(df)
        assert profile.series_profiles["A"]["zero_ratio"] == 0.3

    def test_gap_count(self) -> None:
        """Test gap count metric."""
        dates = [
            "2024-01-01", "2024-01-02", "2024-01-10",  # Gap here
            "2024-01-11", "2024-01-12", "2024-01-20",  # And here
        ]
        df = pd.DataFrame({
            "unique_id": ["A"] * 6,
            "ds": pd.to_datetime(dates),
            "y": [1.0] * 6,
        })
        profile = compute_sparsity_profile(df)
        assert profile.series_profiles["A"]["gap_count"] >= 1

    def test_missing_ratio(self) -> None:
        """Test missing value ratio metric."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        profile = compute_sparsity_profile(df)
        assert profile.series_profiles["A"]["missing_ratio"] == 0.2
