"""Tests for feature versioning and hashing."""

import pytest

from tsagentkit.features.versioning import (
    FeatureConfig,
    compute_feature_hash,
    config_from_dict,
    config_to_dict,
    configs_equal,
)


class TestFeatureConfig:
    """Test FeatureConfig validation."""

    def test_valid_config(self):
        """Test creating a valid feature config."""
        config = FeatureConfig(
            engine="native",
            lags=[1, 7, 14],
            calendar_features=["dayofweek", "month"],
            rolling_windows={7: ["mean", "std"]},
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
        )
        assert config.lags == [1, 7, 14]
        assert config.calendar_features == ["dayofweek", "month"]

    def test_invalid_lag_negative(self):
        """Test that negative lags raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            FeatureConfig(engine="native", lags=[-1, 7])

    def test_invalid_lag_zero(self):
        """Test that zero lag raises ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            FeatureConfig(engine="native", lags=[0, 7])

    def test_invalid_calendar_feature(self):
        """Test that invalid calendar features raise ValueError."""
        with pytest.raises(ValueError, match="Invalid calendar features"):
            FeatureConfig(engine="native", calendar_features=["invalid_feature"])

    def test_invalid_rolling_aggregation(self):
        """Test that invalid rolling aggregations raise ValueError."""
        with pytest.raises(ValueError, match="Invalid aggregations"):
            FeatureConfig(engine="native", rolling_windows={7: ["invalid_agg"]})

    def test_invalid_window_size(self):
        """Test that invalid window sizes raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            FeatureConfig(engine="native", rolling_windows={0: ["mean"]})

    def test_covariate_overlap_raises(self):
        """Test that overlapping known/observed covariates raise ValueError."""
        with pytest.raises(ValueError, match="both known and observed"):
            FeatureConfig(
                engine="native",
                known_covariates=["promo"],
                observed_covariates=["promo"],
            )

    def test_empty_config_valid(self):
        """Test that empty config is valid."""
        config = FeatureConfig(engine="native")
        assert config.lags == []
        assert config.calendar_features == []
        assert config.rolling_windows == {}


class TestComputeFeatureHash:
    """Test feature hash computation."""

    def test_hash_length(self):
        """Test that hash is 16 characters."""
        config = FeatureConfig(engine="native", lags=[1, 7])
        hash_value = compute_feature_hash(config)
        assert len(hash_value) == 16

    def test_deterministic_hash(self):
        """Test that same config produces same hash."""
        config1 = FeatureConfig(engine="native", lags=[1, 7], calendar_features=["dayofweek"])
        config2 = FeatureConfig(engine="native", lags=[1, 7], calendar_features=["dayofweek"])
        assert compute_feature_hash(config1) == compute_feature_hash(config2)

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes."""
        config1 = FeatureConfig(engine="native", lags=[1, 7])
        config2 = FeatureConfig(engine="native", lags=[1, 14])
        assert compute_feature_hash(config1) != compute_feature_hash(config2)

    def test_order_independence(self):
        """Test that hash is independent of order in lists."""
        config1 = FeatureConfig(engine="native", lags=[1, 7, 14], calendar_features=["month", "dayofweek"])
        config2 = FeatureConfig(engine="native", lags=[14, 1, 7], calendar_features=["dayofweek", "month"])
        assert compute_feature_hash(config1) == compute_feature_hash(config2)

    def test_hash_with_all_features(self):
        """Test hash computation with all feature types."""
        config = FeatureConfig(
            engine="native",
            lags=[1, 7, 14],
            calendar_features=["dayofweek", "month", "quarter"],
            rolling_windows={7: ["mean", "std"], 30: ["mean"]},
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
            include_intercept=True,
        )
        hash_value = compute_feature_hash(config)
        assert len(hash_value) == 16
        assert isinstance(hash_value, str)


class TestConfigsEqual:
    """Test config equality comparison."""

    def test_equal_configs(self):
        """Test that identical configs are equal."""
        config1 = FeatureConfig(engine="native", lags=[1, 7])
        config2 = FeatureConfig(engine="native", lags=[1, 7])
        assert configs_equal(config1, config2)

    def test_unequal_configs(self):
        """Test that different configs are not equal."""
        config1 = FeatureConfig(engine="native", lags=[1, 7])
        config2 = FeatureConfig(engine="native", lags=[1, 14])
        assert not configs_equal(config1, config2)


class TestConfigSerialization:
    """Test config serialization to/from dict."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = FeatureConfig(
            engine="native",
            lags=[1, 7],
            calendar_features=["dayofweek"],
            rolling_windows={7: ["mean"]},
            known_covariates=["holiday"],
            include_intercept=True,
        )
        d = config_to_dict(config)
        assert d["engine"] == "native"
        assert d["lags"] == [1, 7]
        assert d["calendar_features"] == ["dayofweek"]
        assert d["rolling_windows"] == {7: ["mean"]}
        assert d["known_covariates"] == ["holiday"]
        assert d["include_intercept"] is True

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "lags": [1, 7],
            "calendar_features": ["dayofweek"],
            "rolling_windows": {7: ["mean"]},
            "known_covariates": ["holiday"],
            "observed_covariates": [],
            "include_intercept": True,
        }
        config = config_from_dict(d)
        assert config.engine == "auto"
        assert config.lags == [1, 7]
        assert config.calendar_features == ["dayofweek"]
        assert config.rolling_windows == {7: ["mean"]}
        assert config.known_covariates == ["holiday"]
        assert config.include_intercept is True

    def test_roundtrip_serialization(self):
        """Test that config survives roundtrip through dict."""
        original = FeatureConfig(
            engine="native",
            lags=[1, 7, 14],
            calendar_features=["dayofweek", "month"],
            rolling_windows={7: ["mean", "std"], 30: ["mean"]},
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
        )
        d = config_to_dict(original)
        restored = config_from_dict(d)
        assert configs_equal(original, restored)
