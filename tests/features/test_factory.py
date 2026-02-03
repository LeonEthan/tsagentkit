"""Tests for FeatureFactory point-in-time safe feature engineering."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.features.factory import FeatureConfig, FeatureFactory
from tsagentkit.features.matrix import FeatureMatrix


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "unique_id": ["A"] * 15 + ["B"] * 15,
        "ds": list(dates[:15]) * 2,
        "y": list(range(1, 16)) + list(range(1, 16)),
    })


@pytest.fixture
def sample_tsdataset():
    """Create a mock TSDataset-like object."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "unique_id": ["A"] * 15 + ["B"] * 15,
        "ds": list(dates[:15]) + list(dates[:15]),
        "y": list(range(1, 16)) + list(range(1, 16)),
    })
    # Mock TSDataset with data attribute
    class MockTSDataset:
        def __init__(self, data):
            self.data = data
    return MockTSDataset(df)


class TestFeatureConfigValidation:
    """Test FeatureConfig validation in factory context."""

    def test_valid_config_creates_factory(self):
        """Test that valid config creates factory successfully."""
        config = FeatureConfig(engine="native", 
            lags=[1, 7],
            calendar_features=["dayofweek"],
            rolling_windows={7: ["mean"]},
        )
        factory = FeatureFactory(config)
        assert factory.config == config

    def test_invalid_config_raises(self):
        """Test that invalid config raises during factory creation."""
        with pytest.raises(ValueError):
            config = FeatureConfig(engine="native", lags=[-1])  # Invalid: negative lag
            FeatureFactory(config)


class TestCreateLagFeatures:
    """Test lag feature creation."""

    def test_single_lag(self, sample_df):
        """Test creating single lag feature."""
        config = FeatureConfig(engine="native", lags=[1])
        factory = FeatureFactory(config)
        result = factory._create_lag_features(sample_df.copy(), [1])

        assert "y_lag_1" in result.columns
        # First row of each series should be null (no previous value)
        series_a = result[result["unique_id"] == "A"]
        assert pd.isna(series_a["y_lag_1"].iloc[0])
        # Second row should have first row's value
        assert series_a["y_lag_1"].iloc[1] == 1

    def test_multiple_lags(self, sample_df):
        """Test creating multiple lag features."""
        config = FeatureConfig(engine="native", lags=[1, 7, 14])
        factory = FeatureFactory(config)
        result = factory._create_lag_features(sample_df.copy(), [1, 7, 14])

        assert "y_lag_1" in result.columns
        assert "y_lag_7" in result.columns
        assert "y_lag_14" in result.columns

    def test_lags_within_series_only(self, sample_df):
        """Test that lags don't cross series boundaries."""
        config = FeatureConfig(engine="native", lags=[1])
        factory = FeatureFactory(config)
        result = factory._create_lag_features(sample_df.copy(), [1])

        # First row of series B should be null, not last value of series A
        series_b_first = result[result["unique_id"] == "B"].iloc[0]
        assert pd.isna(series_b_first["y_lag_1"])


class TestCreateCalendarFeatures:
    """Test calendar feature creation."""

    def test_dayofweek(self, sample_df):
        """Test creating dayofweek feature."""
        config = FeatureConfig(engine="native", calendar_features=["dayofweek"])
        factory = FeatureFactory(config)
        result = factory._create_calendar_features(sample_df.copy(), ["dayofweek"])

        assert "dayofweek" in result.columns
        # 2024-01-01 is a Monday (dayofweek=0)
        assert result["dayofweek"].iloc[0] == 0

    def test_month(self, sample_df):
        """Test creating month feature."""
        config = FeatureConfig(engine="native", calendar_features=["month"])
        factory = FeatureFactory(config)
        result = factory._create_calendar_features(sample_df.copy(), ["month"])

        assert "month" in result.columns
        assert result["month"].iloc[0] == 1  # January

    def test_quarter(self, sample_df):
        """Test creating quarter feature."""
        config = FeatureConfig(engine="native", calendar_features=["quarter"])
        factory = FeatureFactory(config)
        result = factory._create_calendar_features(sample_df.copy(), ["quarter"])

        assert "quarter" in result.columns
        assert result["quarter"].iloc[0] == 1  # Q1

    def test_multiple_calendar_features(self, sample_df):
        """Test creating multiple calendar features."""
        config = FeatureConfig(engine="native", calendar_features=["dayofweek", "month", "year"])
        factory = FeatureFactory(config)
        result = factory._create_calendar_features(
            sample_df.copy(), ["dayofweek", "month", "year"]
        )

        assert "dayofweek" in result.columns
        assert "month" in result.columns
        assert "year" in result.columns
        assert result["year"].iloc[0] == 2024


class TestCreateRollingFeatures:
    """Test rolling window feature creation."""

    def test_rolling_mean(self, sample_df):
        """Test creating rolling mean feature."""
        config = FeatureConfig(engine="native", rolling_windows={7: ["mean"]})
        factory = FeatureFactory(config)
        # Need to sort first for proper rolling calculation
        df = sample_df.sort_values(["unique_id", "ds"])
        result = factory._create_rolling_features(df, {7: ["mean"]})

        assert "y_rolling_mean_7" in result.columns
        # Check that rolling mean is calculated
        series_a = result[result["unique_id"] == "A"]
        # Values should be increasing (1, 2, 3...)
        assert series_a["y_rolling_mean_7"].iloc[6] == pytest.approx(3.5)  # Mean of 1-6

    def test_rolling_std(self, sample_df):
        """Test creating rolling std feature."""
        config = FeatureConfig(engine="native", rolling_windows={7: ["std"]})
        factory = FeatureFactory(config)
        df = sample_df.sort_values(["unique_id", "ds"])
        result = factory._create_rolling_features(df, {7: ["std"]})

        assert "y_rolling_std_7" in result.columns

    def test_multiple_windows(self, sample_df):
        """Test creating features for multiple window sizes."""
        config = FeatureConfig(engine="native", rolling_windows={7: ["mean"], 14: ["mean", "std"]})
        factory = FeatureFactory(config)
        df = sample_df.sort_values(["unique_id", "ds"])
        result = factory._create_rolling_features(
            df, {7: ["mean"], 14: ["mean", "std"]}
        )

        assert "y_rolling_mean_7" in result.columns
        assert "y_rolling_mean_14" in result.columns
        assert "y_rolling_std_14" in result.columns


class TestCreateFeatures:
    """Test the main create_features method."""

    def test_returns_feature_matrix(self, sample_tsdataset):
        """Test that create_features returns FeatureMatrix."""
        config = FeatureConfig(engine="native", lags=[1])
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        assert isinstance(result, FeatureMatrix)
        assert result.config_hash is not None
        assert result.target_col == "y"

    def test_missing_required_column_raises(self, sample_tsdataset):
        """Test that missing columns raise ValueError."""
        # Remove 'y' column
        sample_tsdataset.data = sample_tsdataset.data.drop(columns=["y"])

        config = FeatureConfig(engine="native", )
        factory = FeatureFactory(config)
        with pytest.raises(ValueError, match="Missing required columns"):
            factory.create_features(sample_tsdataset)

    def test_reference_time_filtering(self, sample_tsdataset):
        """Test that reference_time filters data correctly."""
        config = FeatureConfig(engine="native", )
        factory = FeatureFactory(config)

        reference_time = pd.Timestamp("2024-01-10")
        result = factory.create_features(sample_tsdataset, reference_time=reference_time)

        # All dates should be <= reference_time
        assert result.data["ds"].max() <= reference_time

    def test_feature_cols_populated(self, sample_tsdataset):
        """Test that feature_cols is correctly populated."""
        config = FeatureConfig(engine="native", 
            lags=[1, 7],
            calendar_features=["dayofweek"],
        )
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        expected_features = ["y_lag_1", "y_lag_7", "dayofweek"]
        assert result.feature_cols == expected_features

    def test_known_covariates_included(self, sample_tsdataset):
        """Test that known covariates are included as features."""
        sample_tsdataset.data["holiday"] = [0] * 30

        config = FeatureConfig(engine="native", known_covariates=["holiday"])
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        assert "holiday" in result.feature_cols
        assert "holiday" in result.known_covariates

    def test_observed_covariates_lagged(self, sample_tsdataset):
        """Test that observed covariates get lagged features."""
        sample_tsdataset.data["promo"] = [0, 1] * 15

        config = FeatureConfig(engine="native", observed_covariates=["promo"])
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        # Should create lagged version
        assert "promo_lag_1" in result.data.columns
        assert "promo_lag_1" in result.feature_cols

    def test_intercept_added(self, sample_tsdataset):
        """Test that intercept is added when requested."""
        config = FeatureConfig(engine="native", include_intercept=True)
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        assert "intercept" in result.data.columns
        assert result.data["intercept"].unique() == [1.0]
        assert "intercept" in result.feature_cols

    def test_config_hash_consistency(self, sample_tsdataset):
        """Test that same config produces same hash."""
        config = FeatureConfig(engine="native", lags=[1, 7], calendar_features=["dayofweek"])
        factory1 = FeatureFactory(config)
        factory2 = FeatureFactory(config)

        result1 = factory1.create_features(sample_tsdataset)
        result2 = factory2.create_features(sample_tsdataset)

        assert result1.config_hash == result2.config_hash


class TestGetFeatureImportanceTemplate:
    """Test feature importance template generation."""

    def test_template_structure(self):
        """Test that template has correct structure."""
        config = FeatureConfig(engine="native", 
            lags=[1, 7],
            calendar_features=["dayofweek"],
            rolling_windows={7: ["mean", "std"]},
            known_covariates=["holiday"],
            observed_covariates=["promo"],
            include_intercept=True,
        )
        factory = FeatureFactory(config)
        template = factory.get_feature_importance_template()

        assert template["y_lag_1"] == 0.0
        assert template["y_lag_7"] == 0.0
        assert template["dayofweek"] == 0.0
        assert template["y_rolling_mean_7"] == 0.0
        assert template["y_rolling_std_7"] == 0.0
        assert template["holiday"] == 0.0
        assert template["promo_lag_1"] == 0.0
        assert template["intercept"] == 0.0

    def test_empty_config_template(self):
        """Test template with empty config."""
        config = FeatureConfig(engine="native", )
        factory = FeatureFactory(config)
        template = factory.get_feature_importance_template()

        assert template == {}


class TestPointInTimeSafety:
    """Test that features are point-in-time safe (no lookahead)."""

    def test_lags_use_past_only(self, sample_tsdataset):
        """Test that lag features only use past values."""
        config = FeatureConfig(engine="native", lags=[1])
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        # At time t, lag_1 should equal value at t-1
        series_a = result.data[result.data["unique_id"] == "A"]
        for i in range(1, len(series_a)):
            actual_lag = series_a["y_lag_1"].iloc[i]
            expected_lag = series_a["y"].iloc[i - 1]
            assert actual_lag == expected_lag

    def test_rolling_features_right_aligned(self, sample_tsdataset):
        """Test that rolling features are right-aligned (no future info)."""
        config = FeatureConfig(engine="native", rolling_windows={3: ["mean"]})
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        series_a = result.data[result.data["unique_id"] == "A"]
        # At index 2 (3rd point), rolling_mean_3 should use past values only
        expected = series_a["y"].iloc[0:2].mean()
        assert series_a["y_rolling_mean_3"].iloc[2] == pytest.approx(expected)

    def test_observed_covariates_lagged_prevents_leakage(self, sample_tsdataset):
        """Test that observed covariates are lagged to prevent leakage."""
        # Set up observed covariate
        sample_tsdataset.data["weather"] = list(range(30))

        config = FeatureConfig(engine="native", observed_covariates=["weather"])
        factory = FeatureFactory(config)
        result = factory.create_features(sample_tsdataset)

        # At time t, weather_lag_1 should equal weather at t-1
        series_a = result.data[result.data["unique_id"] == "A"]
        for i in range(1, len(series_a)):
            actual = series_a["weather_lag_1"].iloc[i]
            expected = series_a["weather"].iloc[i - 1]
            assert actual == expected

    def test_reference_time_enforces_cutoff(self, sample_tsdataset):
        """Test that reference_time enforces strict cutoff."""
        config = FeatureConfig(engine="native", )
        factory = FeatureFactory(config)

        reference_time = pd.Timestamp("2024-01-10")
        result = factory.create_features(sample_tsdataset, reference_time)

        # No rows should be after reference_time
        assert (result.data["ds"] > reference_time).sum() == 0
