"""Tests for core data structures (TSDataset, CovariateSet).

Tests data container functionality and validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import CovariateSet, ForecastConfig, TSDataset
from tsagentkit.core.errors import EContract


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    return pd.DataFrame({
        "unique_id": ["A"] * 30,
        "ds": pd.date_range("2024-01-01", periods=30),
        "y": range(30),
    })


@pytest.fixture
def multi_series_df():
    """Create multi-series DataFrame."""
    dfs = []
    for uid in ["A", "B", "C"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 20,
            "ds": pd.date_range("2024-01-01", periods=20),
            "y": np.random.randn(20).cumsum() + 10,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def config():
    """Create default config."""
    return ForecastConfig(h=7, freq="D")


class TestCovariateSet:
    """Test CovariateSet data structure."""

    def test_empty_covariate_set(self):
        """Create empty covariate set."""
        cov = CovariateSet()
        assert cov.static is None
        assert cov.past is None
        assert cov.future is None
        assert cov.is_empty()

    def test_covariate_set_with_static(self):
        """Create covariate set with static covariates."""
        static_df = pd.DataFrame({
            "unique_id": ["A", "B"],
            "category": ["X", "Y"],
        })
        cov = CovariateSet(static=static_df)
        assert not cov.is_empty()
        assert cov.static is not None
        assert cov.past is None
        assert cov.future is None

    def test_covariate_set_with_past(self):
        """Create covariate set with past covariates."""
        past_df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "feature": range(10),
        })
        cov = CovariateSet(past=past_df)
        assert not cov.is_empty()
        assert cov.past is not None

    def test_covariate_set_with_future(self):
        """Create covariate set with future covariates."""
        future_df = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-31", periods=7),
            "promotion": [1, 0, 0, 1, 0, 0, 0],
        })
        cov = CovariateSet(future=future_df)
        assert not cov.is_empty()
        assert cov.future is not None

    def test_covariate_set_with_all(self):
        """Create covariate set with all types."""
        static_df = pd.DataFrame({"unique_id": ["A"], "category": ["X"]})
        past_df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "feature": range(10),
        })
        future_df = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-31", periods=7),
            "promotion": [1] * 7,
        })
        cov = CovariateSet(static=static_df, past=past_df, future=future_df)
        assert not cov.is_empty()
        assert cov.static is not None
        assert cov.past is not None
        assert cov.future is not None


class TestTSDatasetCreation:
    """Test TSDataset creation."""

    def test_from_dataframe_basic(self, sample_df, config):
        """Create TSDataset from DataFrame."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        assert dataset.df is not None
        assert dataset.config == config
        assert dataset.covariates is None

    def test_from_dataframe_with_covariates(self, sample_df, config):
        """Create TSDataset with covariates."""
        cov = CovariateSet(static=pd.DataFrame({"unique_id": ["A"], "cat": ["X"]}))
        dataset = TSDataset.from_dataframe(sample_df, config, covariates=cov)
        assert dataset.covariates == cov

    def test_from_dataframe_missing_column(self, sample_df, config):
        """Raise error if column missing."""
        df_missing = sample_df.drop(columns=["y"])
        with pytest.raises(EContract) as exc_info:
            TSDataset.from_dataframe(df_missing, config)
        assert "Missing required columns" in str(exc_info.value)
        assert "y" in str(exc_info.value)

    def test_from_dataframe_missing_multiple_columns(self, config):
        """Raise error if multiple columns missing."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(EContract) as exc_info:
            TSDataset.from_dataframe(df, config)
        assert "unique_id" in str(exc_info.value)
        assert "ds" in str(exc_info.value)
        assert "y" in str(exc_info.value)

    def test_from_dataframe_sorts_by_time(self, config):
        """DataFrame is sorted by unique_id and ds."""
        df = pd.DataFrame({
            "unique_id": ["B", "A", "B", "A"],
            "ds": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"]),
            "y": [4, 2, 3, 1],
        })
        dataset = TSDataset.from_dataframe(df, config)
        # Should be sorted: A-01, A-02, B-01, B-02
        assert list(dataset.df["unique_id"]) == ["A", "A", "B", "B"]
        assert list(dataset.df["y"]) == [1, 2, 3, 4]

    def test_from_dataframe_custom_columns(self):
        """Create TSDataset with custom column names."""
        df = pd.DataFrame({
            "series_id": ["A"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "value": range(10),
        })
        config = ForecastConfig(h=7, freq="D", id_col="series_id", time_col="timestamp", target_col="value")
        dataset = TSDataset.from_dataframe(df, config)
        assert dataset.config.id_col == "series_id"
        assert dataset.config.time_col == "timestamp"
        assert dataset.config.target_col == "value"


class TestTSDatasetProperties:
    """Test TSDataset properties."""

    def test_n_series_single(self, sample_df, config):
        """Count single series."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        assert dataset.n_series == 1

    def test_n_series_multiple(self, multi_series_df, config):
        """Count multiple series."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        assert dataset.n_series == 3

    def test_min_length_single(self, sample_df, config):
        """Get min length for single series."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        assert dataset.min_length == 30

    def test_min_length_multiple(self, config):
        """Get min length for multiple series with different lengths."""
        dfs = []
        for i, uid in enumerate(["A", "B", "C"]):
            df = pd.DataFrame({
                "unique_id": [uid] * (10 + i * 5),
                "ds": pd.date_range("2024-01-01", periods=10 + i * 5),
                "y": range(10 + i * 5),
            })
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        dataset = TSDataset.from_dataframe(df, config)
        assert dataset.min_length == 10
        assert dataset.max_length == 20

    def test_max_length(self, sample_df, config):
        """Get max length."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        assert dataset.max_length == 30


class TestTSDatasetGetSeries:
    """Test getting individual series."""

    def test_get_series(self, multi_series_df, config):
        """Get single series by ID."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        series_a = dataset.get_series("A")
        assert len(series_a) == 20
        assert all(series_a["unique_id"] == "A")

    def test_get_series_returns_copy(self, multi_series_df, config):
        """get_series returns a copy."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        series_a = dataset.get_series("A")
        series_a["y"] = 0
        # Original dataset should be unchanged
        assert dataset.df["y"].iloc[0] != 0

    def test_get_nonexistent_series(self, multi_series_df, config):
        """Get series that doesn't exist returns empty DataFrame."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        series = dataset.get_series("Z")
        assert len(series) == 0


class TestTSDatasetFutureIndex:
    """Test future index generation."""

    def test_future_index_single_series(self, sample_df, config):
        """Generate future index for single series."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        future = dataset.future_index()
        assert len(future) == 7  # config.h
        assert "unique_id" in future.columns
        assert "ds" in future.columns
        assert future["ds"].iloc[0] == pd.Timestamp("2024-01-31")

    def test_future_index_multi_series(self, multi_series_df, config):
        """Generate future index for multiple series."""
        dataset = TSDataset.from_dataframe(multi_series_df, config)
        future = dataset.future_index()
        assert len(future) == 21  # 3 series * 7 horizon
        assert set(future["unique_id"].unique()) == {"A", "B", "C"}

    def test_future_index_custom_h(self, sample_df, config):
        """Generate future index with custom horizon."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        future = dataset.future_index(n_periods=14)
        assert len(future) == 14

    def test_future_index_hourly(self):
        """Generate future index with hourly frequency."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 48,
            "ds": pd.date_range("2024-01-01", periods=48, freq="H"),
            "y": range(48),
        })
        config = ForecastConfig(h=12, freq="H")
        dataset = TSDataset.from_dataframe(df, config)
        future = dataset.future_index()
        assert len(future) == 12
        # Should be hourly periods
        first_diff = future["ds"].iloc[1] - future["ds"].iloc[0]
        assert first_diff == pd.Timedelta(hours=1)


class TestTSDatasetImmutability:
    """Test that TSDataset is frozen/immutable."""

    def test_dataset_is_frozen(self, sample_df, config):
        """TSDataset dataclass is frozen."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        with pytest.raises(AttributeError):
            dataset.df = pd.DataFrame()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
