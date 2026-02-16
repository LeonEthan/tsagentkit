"""Tests for ForecastConfig.

Tests configuration validation, presets, and properties.
"""

from __future__ import annotations

import pytest

from tsagentkit import ForecastConfig


class TestForecastConfigValidation:
    """Test config validation."""

    def test_valid_config(self):
        """Create valid config."""
        config = ForecastConfig(h=7, freq="D")
        assert config.h == 7
        assert config.freq == "D"

    def test_h_must_be_positive(self):
        """h must be positive."""
        with pytest.raises(ValueError, match="h must be positive"):
            ForecastConfig(h=0, freq="D")

    def test_h_negative(self):
        """h cannot be negative."""
        with pytest.raises(ValueError, match="h must be positive"):
            ForecastConfig(h=-1, freq="D")

    def test_n_backtest_windows_negative(self):
        """n_backtest_windows cannot be negative."""
        with pytest.raises(ValueError, match="n_backtest_windows must be non-negative"):
            ForecastConfig(h=7, freq="D", n_backtest_windows=-1)

    def test_min_models_for_ensemble_zero(self):
        """min_models_for_ensemble must be at least 1."""
        with pytest.raises(ValueError, match="min_models_for_ensemble must be at least 1"):
            ForecastConfig(h=7, freq="D", min_models_for_ensemble=0)

    def test_min_models_for_ensemble_negative(self):
        """min_models_for_ensemble cannot be negative."""
        with pytest.raises(ValueError, match="min_models_for_ensemble must be at least 1"):
            ForecastConfig(h=7, freq="D", min_models_for_ensemble=-5)


class TestForecastConfigDefaults:
    """Test config defaults."""

    def test_default_quantiles(self):
        """Default quantiles are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]."""
        config = ForecastConfig(h=7, freq="D")
        assert config.quantiles == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_default_columns(self):
        """Default column names."""
        config = ForecastConfig(h=7, freq="D")
        assert config.id_col == "unique_id"
        assert config.time_col == "ds"
        assert config.target_col == "y"

    def test_default_mode(self):
        """Default mode is 'standard'."""
        config = ForecastConfig(h=7, freq="D")
        assert config.mode == "standard"

    def test_default_tsfm_mode(self):
        """Default tsfm_mode is 'preferred'."""
        config = ForecastConfig(h=7, freq="D")
        assert config.tsfm_mode == "preferred"

    def test_default_backtest_windows(self):
        """Default n_backtest_windows is 5."""
        config = ForecastConfig(h=7, freq="D")
        assert config.n_backtest_windows == 5

    def test_default_min_train_size(self):
        """Default min_train_size is 56."""
        config = ForecastConfig(h=7, freq="D")
        assert config.min_train_size == 56

    def test_default_allow_fallback(self):
        """Default allow_fallback is True."""
        config = ForecastConfig(h=7, freq="D")
        assert config.allow_fallback is True

    def test_default_ensemble_method(self):
        """Default ensemble_method is 'median'."""
        config = ForecastConfig(h=7, freq="D")
        assert config.ensemble_method == "median"

    def test_default_require_all_tsfm(self):
        """Default require_all_tsfm is False."""
        config = ForecastConfig(h=7, freq="D")
        assert config.require_all_tsfm is False


class TestForecastConfigPresets:
    """Test config presets."""

    def test_quick_preset(self):
        """Quick preset configuration."""
        config = ForecastConfig.quick(h=7, freq="D")
        assert config.h == 7
        assert config.freq == "D"
        assert config.mode == "quick"
        assert config.tsfm_mode == "preferred"
        assert config.n_backtest_windows == 2
        assert config.allow_fallback is True

    def test_quick_preset_hourly(self):
        """Quick preset with hourly frequency."""
        config = ForecastConfig.quick(h=24, freq="H")
        assert config.h == 24
        assert config.freq == "H"

    def test_standard_preset(self):
        """Standard preset configuration."""
        config = ForecastConfig.standard(h=7, freq="D")
        assert config.h == 7
        assert config.freq == "D"
        assert config.mode == "standard"
        assert config.tsfm_mode == "required"
        assert config.n_backtest_windows == 5
        assert config.allow_fallback is True

    def test_strict_preset(self):
        """Strict preset configuration."""
        config = ForecastConfig.strict(h=7, freq="D")
        assert config.h == 7
        assert config.freq == "D"
        assert config.mode == "strict"
        assert config.tsfm_mode == "required"
        assert config.n_backtest_windows == 5
        assert config.allow_fallback is False
        assert config.require_all_tsfm is True


class TestSeasonLength:
    """Test season_length property."""

    def test_season_length_daily(self):
        """Daily frequency has season_length 7."""
        config = ForecastConfig(h=7, freq="D")
        assert config.season_length == 7

    def test_season_length_hourly(self):
        """Hourly frequency has season_length 24."""
        config = ForecastConfig(h=7, freq="H")
        assert config.season_length == 24

    def test_season_length_monthly(self):
        """Monthly frequency has season_length 12."""
        config = ForecastConfig(h=7, freq="M")
        assert config.season_length == 12

    def test_season_length_quarterly(self):
        """Quarterly frequency has season_length 4."""
        config = ForecastConfig(h=7, freq="Q")
        assert config.season_length == 4

    def test_season_length_business(self):
        """Business day frequency has season_length 5."""
        config = ForecastConfig(h=7, freq="B")
        assert config.season_length == 5

    def test_season_length_weekly(self):
        """Weekly frequency has season_length 52."""
        config = ForecastConfig(h=7, freq="W")
        assert config.season_length == 52

    def test_season_length_minute(self):
        """Minute frequency has season_length 60."""
        config = ForecastConfig(h=7, freq="min")
        assert config.season_length == 60

    def test_season_length_unknown(self):
        """Unknown frequency returns None."""
        config = ForecastConfig(h=7, freq="X")
        assert config.season_length is None

    def test_season_length_with_multiplier(self):
        """Frequency with multiplier is handled."""
        config = ForecastConfig(h=7, freq="2D")
        assert config.season_length == 7

    def test_season_length_ms_month(self):
        """Month start frequency has season_length 12."""
        config = ForecastConfig(h=7, freq="MS")
        assert config.season_length == 12


class TestWithCovariates:
    """Test with_covariates method."""

    def test_with_static_covariates(self):
        """Add static covariates."""
        config = ForecastConfig(h=7, freq="D")
        config_with_cov = config.with_covariates(static=["category", "region"])
        assert hasattr(config_with_cov, "_extra")
        assert config_with_cov._extra["covariates"]["static"] == ["category", "region"]

    def test_with_past_covariates(self):
        """Add past covariates."""
        config = ForecastConfig(h=7, freq="D")
        config_with_cov = config.with_covariates(past=["feature1", "feature2"])
        assert config_with_cov._extra["covariates"]["past"] == ["feature1", "feature2"]

    def test_with_future_covariates(self):
        """Add future covariates."""
        config = ForecastConfig(h=7, freq="D")
        config_with_cov = config.with_covariates(future=["promotion", "holiday"])
        assert config_with_cov._extra["covariates"]["future"] == ["promotion", "holiday"]

    def test_with_all_covariates(self):
        """Add all covariate types."""
        config = ForecastConfig(h=7, freq="D")
        config_with_cov = config.with_covariates(
            static=["category"],
            past=["feature1"],
            future=["promotion"],
        )
        assert config_with_cov._extra["covariates"]["static"] == ["category"]
        assert config_with_cov._extra["covariates"]["past"] == ["feature1"]
        assert config_with_cov._extra["covariates"]["future"] == ["promotion"]


class TestConfigImmutability:
    """Test that config is frozen/immutable."""

    def test_config_is_frozen(self):
        """Config dataclass is frozen."""
        config = ForecastConfig(h=7, freq="D")
        with pytest.raises(AttributeError):
            config.h = 14

    def test_config_is_frozen_freq(self):
        """Cannot modify freq."""
        config = ForecastConfig(h=7, freq="D")
        with pytest.raises(AttributeError):
            config.freq = "H"


class TestConfigEquality:
    """Test config equality."""

    def test_same_config_equal(self):
        """Same configs are equal."""
        config1 = ForecastConfig(h=7, freq="D")
        config2 = ForecastConfig(h=7, freq="D")
        assert config1 == config2

    def test_different_h_not_equal(self):
        """Different h values are not equal."""
        config1 = ForecastConfig(h=7, freq="D")
        config2 = ForecastConfig(h=14, freq="D")
        assert config1 != config2

    def test_different_freq_not_equal(self):
        """Different freq values are not equal."""
        config1 = ForecastConfig(h=7, freq="D")
        config2 = ForecastConfig(h=7, freq="H")
        assert config1 != config2


class TestConfigRepr:
    """Test config string representation."""

    def test_repr_contains_key_info(self):
        """Repr contains key configuration info."""
        config = ForecastConfig(h=7, freq="D")
        repr_str = repr(config)
        assert "ForecastConfig" in repr_str
        assert "h=7" in repr_str
        assert "freq='D'" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
