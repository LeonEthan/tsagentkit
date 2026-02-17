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

    def test_min_tsfm_zero(self):
        """min_tsfm must be at least 1."""
        with pytest.raises(ValueError, match="min_tsfm must be at least 1"):
            ForecastConfig(h=7, freq="D", min_tsfm=0)

    def test_min_tsfm_negative(self):
        """min_tsfm cannot be negative."""
        with pytest.raises(ValueError, match="min_tsfm must be at least 1"):
            ForecastConfig(h=7, freq="D", min_tsfm=-5)


class TestForecastConfigDefaults:
    """Test config defaults."""

    def test_default_quantiles(self):
        """Default quantiles are (0.1, 0.5, 0.9)."""
        config = ForecastConfig(h=7, freq="D")
        assert config.quantiles == (0.1, 0.5, 0.9)

    def test_default_ensemble_method(self):
        """Default ensemble_method is 'median'."""
        config = ForecastConfig(h=7, freq="D")
        assert config.ensemble_method == "median"

    def test_default_min_tsfm(self):
        """Default min_tsfm is 1."""
        config = ForecastConfig(h=7, freq="D")
        assert config.min_tsfm == 1


class TestForecastConfigPresets:
    """Test config presets."""

    def test_quick_preset(self):
        """Quick preset configuration."""
        config = ForecastConfig.quick(h=7, freq="D")
        assert config.h == 7
        assert config.freq == "D"

    def test_quick_preset_hourly(self):
        """Quick preset with hourly frequency."""
        config = ForecastConfig.quick(h=24, freq="H")
        assert config.h == 24
        assert config.freq == "H"

    def test_strict_preset(self):
        """Strict preset configuration."""
        config = ForecastConfig.strict(h=7, freq="D")
        assert config.h == 7
        assert config.freq == "D"
        assert config.min_tsfm == 1


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
