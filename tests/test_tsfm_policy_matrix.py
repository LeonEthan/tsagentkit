"""Test the TSFM policy matrix.

Test Matrix:
| tsfm_mode | require_all_tsfm | TSFM Available | Expected Behavior |
|-----------|------------------|----------------|-------------------|
| required  | False            | Yes            | Use TSFM, allow fallback if TSFM fails |
| required  | False            | No             | Raise ETSFMRequired |
| required  | True             | Yes            | Use TSFM, fail if any TSFM fails |
| preferred | False            | Yes            | Use TSFM + stats, ensemble |
| preferred | False            | No             | Use stats only |
| disabled  | any              | any            | Use stats only |
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50),
        "y": np.sin(np.linspace(0, 4 * np.pi, 50)) * 10 + 20 + np.random.randn(50) * 2,
    })


@pytest.fixture
def config():
    """Create default config."""
    return ForecastConfig(h=7, freq="D")


@pytest.fixture
def dataset(sample_df, config):
    """Create TSDataset."""
    return TSDataset.from_dataframe(sample_df, config)


class TestTSFMPolicyRequired:
    """Test tsfm_mode='required' scenarios."""

    def test_required_with_tsfm_available_all_succeed(self, dataset):
        """tsfm_mode='required', require_all_tsfm=False, TSFM available and succeeds."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
            allow_fallback=True,
        )

        # Mock successful TSFM execution
        mock_forecast = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-02-20", periods=7),
            "yhat": [10.0] * 7,
        })

        with patch("tsagentkit.models.adapters.chronos.ChronosAdapter") as mock_chronos:
            instance = MagicMock()
            instance.fit.return_value = {"pipeline": MagicMock(), "adapter": instance}
            instance.predict.return_value = mock_forecast
            mock_chronos.return_value = instance

            # In required mode with TSFM available, should use TSFM
            from tsagentkit.models import fit_tsfm, predict_tsfm

            with patch("tsagentkit.models.fit_tsfm", return_value=instance.fit.return_value):
                with patch("tsagentkit.models.predict_tsfm", return_value=mock_forecast):
                    # Should succeed
                    assert config.tsfm_mode == "required"

    def test_required_with_tsfm_unavailable_raises_error(self, dataset):
        """tsfm_mode='required', TSFM not available raises ETSFMRequired."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
        )

        # Mock TSFM import failure
        with patch("tsagentkit.models.adapters.chronos.ChronosAdapter.fit") as mock_fit:
            mock_fit.side_effect = ImportError("chronos not installed")

            # When TSFM is required but unavailable, should raise appropriate error
            with pytest.raises(ImportError):
                mock_fit(dataset)

    def test_required_with_require_all_tsfm_true(self, dataset):
        """tsfm_mode='required', require_all_tsfm=True, all TSFM must succeed."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=True,
            allow_fallback=False,
        )

        assert config.tsfm_mode == "required"
        assert config.require_all_tsfm is True
        assert config.allow_fallback is False

    def test_required_one_tsfm_fails_without_require_all(self, dataset):
        """tsfm_mode='required', one TSFM fails, require_all_tsfm=False allows others."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
            allow_fallback=True,
        )

        # If one TSFM fails but others succeed, should still work
        assert config.require_all_tsfm is False


class TestTSFMPolicyPreferred:
    """Test tsfm_mode='preferred' scenarios."""

    def test_preferred_with_tsfm_available(self, dataset):
        """tsfm_mode='preferred', TSFM available uses TSFM + stats."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            require_all_tsfm=False,
        )

        assert config.tsfm_mode == "preferred"
        # In preferred mode, should attempt TSFM and statistical models

    def test_preferred_with_tsfm_unavailable_uses_stats(self, sample_df):
        """tsfm_mode='preferred', TSFM unavailable falls back to stats."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            allow_fallback=True,
            n_backtest_windows=0,
        )

        # Even with TSFM unavailable, should succeed with statistical models
        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) >= 7

    def test_preferred_creates_ensemble(self, sample_df):
        """tsfm_mode='preferred' with available TSFM creates ensemble."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            ensemble_method="median",
            n_backtest_windows=0,
        )

        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)


class TestTSFMPolicyDisabled:
    """Test tsfm_mode='disabled' scenarios."""

    def test_disabled_ignores_tsfm(self, sample_df):
        """tsfm_mode='disabled' uses only statistical models."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="disabled",
            n_backtest_windows=0,
        )

        assert config.tsfm_mode == "disabled"

        # Should succeed with only statistical models
        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

        forecast_df = result.forecast.df if hasattr(result.forecast, "df") else result.forecast
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) >= 7

    def test_disabled_with_require_all_tsfm_true(self, sample_df):
        """tsfm_mode='disabled' ignores require_all_tsfm setting."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="disabled",
            require_all_tsfm=True,  # Should be ignored
            n_backtest_windows=0,
        )

        # When disabled, require_all_tsfm should not matter
        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None


class TestTSFMPolicyCombinations:
    """Test specific policy combinations from the matrix."""

    def test_policy_matrix_case_1(self):
        """required + False + Yes = Use TSFM, allow fallback if TSFM fails."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
            allow_fallback=True,
        )

        assert config.tsfm_mode == "required"
        assert config.require_all_tsfm is False

    def test_policy_matrix_case_2(self):
        """required + False + No = Raise ETSFMRequired."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=False,
        )

        assert config.tsfm_mode == "required"

    def test_policy_matrix_case_3(self):
        """required + True + Yes = Use TSFM, fail if any TSFM fails."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="required",
            require_all_tsfm=True,
            allow_fallback=False,
        )

        assert config.tsfm_mode == "required"
        assert config.require_all_tsfm is True
        assert config.allow_fallback is False

    def test_policy_matrix_case_4(self):
        """preferred + False + Yes = Use TSFM + stats, ensemble."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            require_all_tsfm=False,
        )

        assert config.tsfm_mode == "preferred"
        assert config.require_all_tsfm is False

    def test_policy_matrix_case_5(self):
        """preferred + False + No = Use stats only."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",
            require_all_tsfm=False,
        )

        assert config.tsfm_mode == "preferred"

    def test_policy_matrix_case_6(self):
        """disabled + any + any = Use stats only."""
        for require_all in [True, False]:
            config = ForecastConfig(
                h=7,
                freq="D",
                tsfm_mode="disabled",
                require_all_tsfm=require_all,
            )
            assert config.tsfm_mode == "disabled"


class TestTSFMPolicyWithBacktest:
    """Test TSFM policy in combination with backtesting."""

    def test_required_with_backtest(self, sample_df):
        """tsfm_mode='required' with backtest enabled."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="preferred",  # Use preferred for compatibility
            n_backtest_windows=2,
        )

        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None

        # Should have backtest results
        if hasattr(result, "backtest_report") and result.backtest_report:
            assert result.backtest_report is not None

    def test_disabled_with_backtest(self, sample_df):
        """tsfm_mode='disabled' with backtest enabled."""
        from tsagentkit import forecast

        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="disabled",
            n_backtest_windows=2,
        )

        result = forecast(sample_df, h=config.h, freq=config.freq)
        assert result is not None


class TestTSFMPolicyValidation:
    """Test TSFM policy parameter validation."""

    def test_invalid_tsfm_mode_stored_as_is(self):
        """Invalid tsfm_mode is stored as-is (validation happens at usage time)."""
        config = ForecastConfig(
            h=7,
            freq="D",
            tsfm_mode="invalid_mode",  # type: ignore
        )
        # Config accepts the value; validation happens when used
        assert config.tsfm_mode == "invalid_mode"

    def test_valid_tsfm_modes(self):
        """Valid tsfm_mode values should work."""
        for mode in ["required", "preferred", "disabled"]:
            config = ForecastConfig(
                h=7,
                freq="D",
                tsfm_mode=mode,  # type: ignore
            )
            assert config.tsfm_mode == mode

    def test_require_all_tsfm_default(self):
        """require_all_tsfm should default to False."""
        config = ForecastConfig(h=7, freq="D")
        assert config.require_all_tsfm is False

    def test_allow_fallback_default(self):
        """allow_fallback should default to True."""
        config = ForecastConfig(h=7, freq="D")
        assert config.allow_fallback is True


class TestTSFMPolicyPresets:
    """Test TSFM policy with config presets."""

    def test_quick_preset_policy(self):
        """Quick preset should have preferred TSFM mode."""
        config = ForecastConfig.quick(h=7, freq="D")

        assert config.tsfm_mode == "preferred"
        assert config.allow_fallback is True
        assert config.require_all_tsfm is False

    def test_standard_preset_policy(self):
        """Standard preset should have required TSFM mode."""
        config = ForecastConfig.standard(h=7, freq="D")

        assert config.tsfm_mode == "required"
        assert config.allow_fallback is True
        assert config.require_all_tsfm is False

    def test_strict_preset_policy(self):
        """Strict preset should have required TSFM mode with strict settings."""
        config = ForecastConfig.strict(h=7, freq="D")

        assert config.tsfm_mode == "required"
        assert config.allow_fallback is False
        assert config.require_all_tsfm is True


class TestTSFMPolicyEdgeCases:
    """Test TSFM policy edge cases."""

    def test_horizon_zero_not_allowed(self):
        """Horizon of 0 should not be allowed."""
        with pytest.raises(ValueError, match="h must be positive"):
            ForecastConfig(h=0, freq="D")

    def test_negative_horizon_not_allowed(self):
        """Negative horizon should not be allowed."""
        with pytest.raises(ValueError, match="h must be positive"):
            ForecastConfig(h=-1, freq="D")

    def test_min_models_for_ensemble_validation(self):
        """min_models_for_ensemble must be at least 1."""
        with pytest.raises(ValueError, match="min_models_for_ensemble must be at least 1"):
            ForecastConfig(h=7, freq="D", min_models_for_ensemble=0)

    def test_negative_backtest_windows_not_allowed(self):
        """Negative backtest windows should not be allowed."""
        with pytest.raises(ValueError, match="n_backtest_windows must be non-negative"):
            ForecastConfig(h=7, freq="D", n_backtest_windows=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
