"""Tests for router bucketing (Head/Tail, Short/Long history)."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.router.bucketing import (
    BucketConfig,
    BucketProfile,
    BucketStatistics,
    DataBucketer,
    SeriesBucket,
)


class TestSeriesBucket:
    """Test SeriesBucket enum."""

    def test_bucket_values(self):
        """Test bucket enum values."""
        assert SeriesBucket.HEAD.value == "head"
        assert SeriesBucket.TAIL.value == "tail"
        assert SeriesBucket.SHORT_HISTORY.value == "short_history"
        assert SeriesBucket.LONG_HISTORY.value == "long_history"


class TestBucketConfig:
    """Test BucketConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = BucketConfig()
        assert config.head_quantile_threshold == 0.8
        assert config.tail_quantile_threshold == 0.2
        assert config.short_history_max_obs == 30
        assert config.long_history_min_obs == 365

    def test_custom_config(self):
        """Test custom configuration."""
        config = BucketConfig(
            head_quantile_threshold=0.9,
            short_history_max_obs=50,
        )
        assert config.head_quantile_threshold == 0.9
        assert config.short_history_max_obs == 50

    def test_invalid_head_threshold(self):
        """Test invalid head threshold raises error."""
        with pytest.raises(ValueError, match="head_quantile_threshold"):
            BucketConfig(head_quantile_threshold=1.5)

    def test_invalid_tail_threshold(self):
        """Test invalid tail threshold raises error."""
        with pytest.raises(ValueError, match="tail_quantile_threshold"):
            BucketConfig(tail_quantile_threshold=-0.1)

    def test_tail_greater_than_head(self):
        """Test tail > head raises error."""
        with pytest.raises(ValueError, match="tail_quantile_threshold"):
            BucketConfig(
                tail_quantile_threshold=0.6,
                head_quantile_threshold=0.5,
            )

    def test_invalid_history_thresholds(self):
        """Test invalid history thresholds raise error."""
        with pytest.raises(ValueError, match="long_history_min_obs"):
            BucketConfig(
                short_history_max_obs=100,
                long_history_min_obs=50,
            )


class TestBucketStatistics:
    """Test BucketStatistics dataclass."""

    def test_creation(self):
        """Test creating statistics."""
        stats = BucketStatistics(
            series_count=10,
            total_observations=1000,
            avg_observations=100.0,
            avg_value=50.0,
            value_percentile=0.25,
        )
        assert stats.series_count == 10
        assert stats.total_observations == 1000


class TestBucketProfile:
    """Test BucketProfile dataclass."""

    @pytest.fixture
    def sample_profile(self):
        """Create sample bucket profile."""
        return BucketProfile(
            bucket_assignments={
                "A": {SeriesBucket.HEAD, SeriesBucket.LONG_HISTORY},
                "B": {SeriesBucket.TAIL, SeriesBucket.SHORT_HISTORY},
                "C": set(),  # Middle series
            },
            bucket_stats={},
        )

    def test_get_bucket_for_series(self, sample_profile):
        """Test getting bucket for series."""
        assert sample_profile.get_bucket_for_series("A") == {
            SeriesBucket.HEAD, SeriesBucket.LONG_HISTORY
        }
        assert sample_profile.get_bucket_for_series("B") == {
            SeriesBucket.TAIL, SeriesBucket.SHORT_HISTORY
        }
        assert sample_profile.get_bucket_for_series("C") == set()
        assert sample_profile.get_bucket_for_series("D") == set()  # Not found

    def test_get_series_in_bucket(self, sample_profile):
        """Test getting series in bucket."""
        head_series = sample_profile.get_series_in_bucket(SeriesBucket.HEAD)
        assert "A" in head_series
        assert "B" not in head_series

    def test_get_bucket_counts(self, sample_profile):
        """Test getting bucket counts."""
        counts = sample_profile.get_bucket_counts()
        assert counts[SeriesBucket.HEAD] == 1
        assert counts[SeriesBucket.TAIL] == 1
        assert counts[SeriesBucket.LONG_HISTORY] == 1
        assert counts[SeriesBucket.SHORT_HISTORY] == 1

    def test_summary(self, sample_profile):
        """Test summary method."""
        summary = sample_profile.summary()
        assert "Bucket Profile Summary:" in summary
        assert "head:" in summary


class TestDataBucketerInit:
    """Test DataBucketer initialization."""

    def test_default_config(self):
        """Test bucketer with default config."""
        bucketer = DataBucketer()
        assert bucketer.config.head_quantile_threshold == 0.8

    def test_custom_config(self):
        """Test bucketer with custom config."""
        config = BucketConfig(head_quantile_threshold=0.9)
        bucketer = DataBucketer(config)
        assert bucketer.config.head_quantile_threshold == 0.9


class TestBucketByVolume:
    """Test volume-based bucketing."""

    @pytest.fixture
    def volume_df(self):
        """Create DataFrame with varying volumes."""
        np.random.seed(42)
        return pd.DataFrame({
            "unique_id": ["A"] * 100 + ["B"] * 100 + ["C"] * 100 + ["D"] * 100 + ["E"] * 100,
            "y": (
                np.random.normal(1000, 100, 100).tolist() +  # High volume (HEAD)
                np.random.normal(800, 80, 100).tolist() +   # Medium-high
                np.random.normal(500, 50, 100).tolist() +   # Medium
                np.random.normal(200, 20, 100).tolist() +   # Medium-low
                np.random.normal(50, 5, 100).tolist()       # Low volume (TAIL)
            ),
        })

    def test_head_identification(self, volume_df):
        """Test HEAD bucket identification."""
        config = BucketConfig(head_quantile_threshold=0.8)
        bucketer = DataBucketer(config)
        buckets = bucketer.bucket_by_volume(volume_df)

        # Top 20% should be HEAD (1 series out of 5)
        assert "A" in buckets
        assert buckets["A"] == SeriesBucket.HEAD

    def test_tail_identification(self, volume_df):
        """Test TAIL bucket identification."""
        config = BucketConfig(tail_quantile_threshold=0.2)
        bucketer = DataBucketer(config)
        buckets = bucketer.bucket_by_volume(volume_df)

        # Bottom 20% should be TAIL (1 series out of 5)
        assert "E" in buckets
        assert buckets["E"] == SeriesBucket.TAIL

    def test_middle_series_not_bucketed(self, volume_df):
        """Test that middle series are not assigned to volume buckets."""
        config = BucketConfig(head_quantile_threshold=0.8, tail_quantile_threshold=0.2)
        bucketer = DataBucketer(config)
        buckets = bucketer.bucket_by_volume(volume_df)

        # Middle series (B, C, D) should not be in buckets
        assert "B" not in buckets
        assert "C" not in buckets
        assert "D" not in buckets

    def test_missing_value_column(self):
        """Test that missing value column raises error."""
        df = pd.DataFrame({"unique_id": ["A"], "other": [1]})
        bucketer = DataBucketer()

        with pytest.raises(ValueError, match="Value column"):
            bucketer.bucket_by_volume(df)


class TestBucketByHistoryLength:
    """Test history length bucketing."""

    @pytest.fixture
    def history_df(self):
        """Create DataFrame with varying history lengths."""
        return pd.DataFrame({
            "unique_id": (
                ["short"] * 10 +      # 10 observations
                ["medium"] * 100 +    # 100 observations
                ["long"] * 500        # 500 observations
            ),
            "y": [1.0] * 610,
        })

    def test_short_history(self, history_df):
        """Test SHORT_HISTORY identification."""
        config = BucketConfig(short_history_max_obs=30)
        bucketer = DataBucketer(config)
        buckets = bucketer.bucket_by_history_length(history_df)

        assert "short" in buckets
        assert buckets["short"] == SeriesBucket.SHORT_HISTORY

    def test_long_history(self, history_df):
        """Test LONG_HISTORY identification."""
        config = BucketConfig(long_history_min_obs=365)
        bucketer = DataBucketer(config)
        buckets = bucketer.bucket_by_history_length(history_df)

        assert "long" in buckets
        assert buckets["long"] == SeriesBucket.LONG_HISTORY

    def test_medium_history_not_bucketed(self, history_df):
        """Test that medium history is not assigned."""
        config = BucketConfig(short_history_max_obs=30, long_history_min_obs=365)
        bucketer = DataBucketer(config)
        buckets = bucketer.bucket_by_history_length(history_df)

        assert "medium" not in buckets  # Between thresholds


class TestCreateBucketProfile:
    """Test comprehensive bucket profile creation."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        class MockTSDataset:
            def __init__(self, data):
                self.data = data

        df = pd.DataFrame({
            "unique_id": ["A"] * 400 + ["B"] * 400 + ["C"] * 50 + ["D"] * 50,
            "y": (
                np.random.normal(1000, 100, 400).tolist() +
                np.random.normal(100, 10, 400).tolist() +
                np.random.normal(500, 50, 50).tolist() +
                np.random.normal(500, 50, 50).tolist()
            ),
        })
        return MockTSDataset(df)

    def test_profile_creation(self, sample_dataset):
        """Test creating bucket profile."""
        config = BucketConfig(
            head_quantile_threshold=0.8,
            tail_quantile_threshold=0.2,
            short_history_max_obs=100,
            long_history_min_obs=300,
        )
        bucketer = DataBucketer(config)
        profile = bucketer.create_bucket_profile(sample_dataset)

        assert isinstance(profile, BucketProfile)
        assert len(profile.bucket_assignments) == 4

    def test_combined_buckets(self, sample_dataset):
        """Test that series can have multiple buckets."""
        config = BucketConfig(
            head_quantile_threshold=0.8,
            long_history_min_obs=300,
        )
        bucketer = DataBucketer(config)
        profile = bucketer.create_bucket_profile(sample_dataset)

        # Series A should be both HEAD and LONG_HISTORY
        buckets = profile.get_bucket_for_series("A")
        assert SeriesBucket.HEAD in buckets
        assert SeriesBucket.LONG_HISTORY in buckets


class TestGetModelForBucket:
    """Test model recommendations per bucket."""

    def test_head_recommendation(self):
        """Test model recommendation for HEAD bucket."""
        bucketer = DataBucketer()
        model = bucketer.get_model_for_bucket(SeriesBucket.HEAD)
        assert model == "SeasonalNaive"  # Placeholder for TSFM

    def test_tail_recommendation(self):
        """Test model recommendation for TAIL bucket."""
        bucketer = DataBucketer()
        model = bucketer.get_model_for_bucket(SeriesBucket.TAIL)
        assert model == "HistoricAverage"

    def test_short_history_recommendation(self):
        """Test model recommendation for SHORT_HISTORY bucket."""
        bucketer = DataBucketer()
        model = bucketer.get_model_for_bucket(SeriesBucket.SHORT_HISTORY)
        assert model == "HistoricAverage"

    def test_long_history_recommendation(self):
        """Test model recommendation for LONG_HISTORY bucket."""
        bucketer = DataBucketer()
        model = bucketer.get_model_for_bucket(SeriesBucket.LONG_HISTORY)
        assert model == "SeasonalNaive"

    def test_intermittent_override(self):
        """Test intermittent sparsity class overrides bucket."""
        bucketer = DataBucketer()
        model = bucketer.get_model_for_bucket(SeriesBucket.HEAD, sparsity_class="intermittent")
        assert model == "Croston"


class TestGetBucketSpecificPlanConfig:
    """Test bucket-specific plan configurations."""

    def test_head_config(self):
        """Test HEAD bucket configuration."""
        bucketer = DataBucketer()
        config = bucketer.get_bucket_specific_plan_config(SeriesBucket.HEAD)
        assert config["model_complexity"] == "high"
        assert config["hyperparameter_tuning"] is True

    def test_tail_config(self):
        """Test TAIL bucket configuration."""
        bucketer = DataBucketer()
        config = bucketer.get_bucket_specific_plan_config(SeriesBucket.TAIL)
        assert config["model_complexity"] == "low"
        assert config["hyperparameter_tuning"] is False

    def test_short_history_config(self):
        """Test SHORT_HISTORY bucket configuration."""
        bucketer = DataBucketer()
        config = bucketer.get_bucket_specific_plan_config(SeriesBucket.SHORT_HISTORY)
        assert config["use_global_model"] is True

    def test_long_history_config(self):
        """Test LONG_HISTORY bucket configuration."""
        bucketer = DataBucketer()
        config = bucketer.get_bucket_specific_plan_config(SeriesBucket.LONG_HISTORY)
        assert config["use_global_model"] is False
