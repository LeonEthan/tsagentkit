"""Tests for covariate management with leakage protection."""

import pandas as pd
import pytest

from tsagentkit.contracts.errors import ECovariateLeakage
from tsagentkit.features.covariates import (
    CovariateConfig,
    CovariateManager,
    CovariatePolicy,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "unique_id": ["A", "A", "A", "B", "B", "B"],
        "ds": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-01", "2024-01-02", "2024-01-03",
        ]),
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "promotion": [0, 1, 0, 1, 0, 1],
        "holiday": [0, 0, 1, 0, 0, 1],
    })


class TestCovariateConfig:
    """Test CovariateConfig dataclass."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = CovariateConfig(
            known=["holiday"],
            observed=["promotion"],
        )
        assert config.known == ["holiday"]
        assert config.observed == ["promotion"]

    def test_overlap_raises(self):
        """Test that overlapping covariates raise error."""
        with pytest.raises(ValueError, match="both known and observed"):
            CovariateConfig(
                known=["promo"],
                observed=["promo"],
            )

    def test_get_policy_known(self):
        """Test getting policy for known covariate."""
        config = CovariateConfig(known=["holiday"], observed=["promo"])
        assert config.get_policy("holiday") == CovariatePolicy.KNOWN

    def test_get_policy_observed(self):
        """Test getting policy for observed covariate."""
        config = CovariateConfig(known=["holiday"], observed=["promo"])
        assert config.get_policy("promo") == CovariatePolicy.OBSERVED

    def test_get_policy_none(self):
        """Test getting policy for non-covariate column."""
        config = CovariateConfig(known=["holiday"])
        assert config.get_policy("other") is None

    def test_all_covariates(self):
        """Test getting all covariate names."""
        config = CovariateConfig(known=["a"], observed=["b", "c"])
        assert config.all_covariates() == ["a", "b", "c"]


class TestCovariateManagerInit:
    """Test CovariateManager initialization."""

    def test_init_with_lists(self):
        """Test initialization with covariate lists."""
        manager = CovariateManager(
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
        )
        assert manager.known_covariates == ["holiday"]
        assert manager.observed_covariates == ["promotion"]

    def test_init_with_none(self):
        """Test initialization with None defaults to empty lists."""
        manager = CovariateManager()
        assert manager.known_covariates == []
        assert manager.observed_covariates == []

    def test_overlap_raises(self):
        """Test that overlapping covariates raise error."""
        with pytest.raises(ValueError, match="both known and observed"):
            CovariateManager(
                known_covariates=["promo"],
                observed_covariates=["promo"],
            )


class TestValidateForPrediction:
    """Test validation to prevent covariate leakage."""

    def test_no_observed_covariates_passes(self, sample_df):
        """Test that validation passes with no observed covariates."""
        manager = CovariateManager()
        # Should not raise
        manager.validate_for_prediction(
            sample_df,
            forecast_start=pd.Timestamp("2024-01-02"),
            horizon=7,
        )

    def test_no_future_values_passes(self, sample_df):
        """Test validation passes when observed covariates have no future values."""
        # Remove future promotion values
        df = sample_df.copy()
        df.loc[df["ds"] >= pd.Timestamp("2024-01-02"), "promotion"] = None

        manager = CovariateManager(observed_covariates=["promotion"])
        # Should not raise
        manager.validate_for_prediction(
            df,
            forecast_start=pd.Timestamp("2024-01-02"),
            horizon=7,
        )

    def test_future_values_raises(self, sample_df):
        """Test validation raises when observed covariates have future values."""
        manager = CovariateManager(observed_covariates=["promotion"])
        with pytest.raises(ECovariateLeakage) as exc_info:
            manager.validate_for_prediction(
                sample_df,
                forecast_start=pd.Timestamp("2024-01-02"),
                horizon=7,
            )
        assert "promotion" in str(exc_info.value)
        assert "future" in str(exc_info.value).lower() or "forecast" in str(exc_info.value).lower()

    def test_future_values_error_context(self, sample_df):
        """Test that error includes proper context."""
        manager = CovariateManager(observed_covariates=["promotion"])
        with pytest.raises(ECovariateLeakage) as exc_info:
            manager.validate_for_prediction(
                sample_df,
                forecast_start=pd.Timestamp("2024-01-02"),
                horizon=7,
            )
        error = exc_info.value
        assert error.context["covariate"] == "promotion"
        assert error.context["future_values_count"] == 4  # 2 dates x 2 series


class TestCreateLaggedObservedFeatures:
    """Test creating lagged features for observed covariates."""

    def test_no_observed_returns_copy(self, sample_df):
        """Test that method returns copy when no observed covariates."""
        manager = CovariateManager()
        result = manager.create_lagged_observed_features(sample_df, lags=[1])
        assert result is not sample_df
        pd.testing.assert_frame_equal(result, sample_df)

    def test_create_single_lag(self, sample_df):
        """Test creating single lag for observed covariate."""
        manager = CovariateManager(observed_covariates=["promotion"])
        result = manager.create_lagged_observed_features(sample_df, lags=[1])

        # Check lag column was created
        assert "promotion_lag_1" in result.columns

        # Check lag is correct (shifted by 1 within each series)
        series_a = result[result["unique_id"] == "A"]
        assert series_a["promotion_lag_1"].iloc[0] != series_a["promotion_lag_1"].iloc[0] or pd.isna(series_a["promotion_lag_1"].iloc[0])

    def test_create_multiple_lags(self, sample_df):
        """Test creating multiple lags for observed covariate."""
        manager = CovariateManager(observed_covariates=["promotion"])
        result = manager.create_lagged_observed_features(sample_df, lags=[1, 2])

        assert "promotion_lag_1" in result.columns
        assert "promotion_lag_2" in result.columns

    def test_missing_column_skipped(self, sample_df):
        """Test that missing columns are skipped gracefully."""
        manager = CovariateManager(observed_covariates=["nonexistent"])
        result = manager.create_lagged_observed_features(sample_df, lags=[1])
        # Should not raise, just skip
        assert "nonexistent_lag_1" not in result.columns


class TestSeparateCovariatesForPrediction:
    """Test separating covariates by type for prediction."""

    def test_separate_known_and_observed(self, sample_df):
        """Test separating known and observed covariates."""
        manager = CovariateManager(
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
        )
        known_df, observed_df = manager.separate_covariates_for_prediction(
            sample_df,
            forecast_start=pd.Timestamp("2024-01-02"),
        )

        assert "holiday" in known_df.columns
        assert "promotion" not in known_df.columns
        assert "promotion" in observed_df.columns
        assert "holiday" not in observed_df.columns

    def test_no_covariates_returns_empty(self, sample_df):
        """Test that empty covariates returns empty DataFrames."""
        manager = CovariateManager()
        known_df, observed_df = manager.separate_covariates_for_prediction(
            sample_df,
            forecast_start=pd.Timestamp("2024-01-02"),
        )

        assert known_df.empty
        assert observed_df.empty


class TestGetConfig:
    """Test getting covariate configuration."""

    def test_get_config(self):
        """Test that get_config returns correct configuration."""
        manager = CovariateManager(
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
        )
        config = manager.get_config()
        assert config.known == ["holiday"]
        assert config.observed == ["promotion"]
