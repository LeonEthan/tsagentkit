"""Tests for consolidated pipeline.

Tests the new consolidated pipeline functions in pipeline.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.core.dataset import CovariateSet
from tsagentkit.core.errors import EContract, EInsufficient
from tsagentkit.pipeline import (
    build_dataset,
    ensemble,
    forecast,
    make_plan,
    run_forecast,
    validate,
)


@pytest.fixture
def sample_df():
    """Create valid sample DataFrame."""
    return pd.DataFrame({
        "unique_id": ["A"] * 30,
        "ds": pd.date_range("2024-01-01", periods=30),
        "y": range(30),
    })


@pytest.fixture
def config():
    """Create default config."""
    return ForecastConfig(h=7, freq="D")


class TestValidate:
    """Test validate function."""

    def test_valid_data_passes(self, sample_df, config):
        """Valid data passes validation."""
        result = validate(sample_df, config)
        assert result is not None
        assert len(result) == 30

    def test_missing_column_raises(self, sample_df, config):
        """Missing column raises EContract."""
        df_missing = sample_df.drop(columns=["y"])
        with pytest.raises(EContract) as exc_info:
            validate(df_missing, config)
        assert "Missing required columns" in str(exc_info.value)

    def test_empty_dataframe_raises(self, config):
        """Empty DataFrame raises EContract."""
        empty_df = pd.DataFrame(columns=["unique_id", "ds", "y"])
        with pytest.raises(EContract, match="empty"):
            validate(empty_df, config)

    def test_null_in_key_columns_raises(self, sample_df, config):
        """Null in key columns raises EContract."""
        for col in ["unique_id", "ds", "y"]:
            df_with_null = sample_df.copy()
            df_with_null.loc[0, col] = None
            with pytest.raises(EContract, match="null"):
                validate(df_with_null, config)

    def test_column_renaming(self, config):
        """Custom columns are renamed to standard."""
        df = pd.DataFrame({
            "series_id": ["A"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "value": range(10),
        })
        custom_config = ForecastConfig(
            h=7, freq="D",
            id_col="series_id", time_col="timestamp", target_col="value"
        )
        result = validate(df, custom_config)
        # Should be renamed to standard column names
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns

    def test_returns_copy(self, sample_df, config):
        """validate returns a copy of the DataFrame."""
        result = validate(sample_df, config)
        result["y"] = 0
        # Original should be unchanged
        assert sample_df["y"].iloc[0] == 0


class TestBuildDataset:
    """Test build_dataset function."""

    def test_build_dataset(self, sample_df, config):
        """Build TSDataset from DataFrame."""
        dataset = build_dataset(sample_df, config)
        assert isinstance(dataset, TSDataset)
        assert dataset.config == config

    def test_build_dataset_with_covariates(self, sample_df, config):
        """Build TSDataset with covariates."""
        cov = CovariateSet(static=pd.DataFrame({"unique_id": ["A"], "cat": ["X"]}))
        dataset = build_dataset(sample_df, config, covariates=cov)
        assert dataset.covariates == cov


class TestMakePlan:
    """Test make_plan function."""

    def test_make_plan_returns_models(self, sample_df, config):
        """make_plan returns list of models."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        models = make_plan(dataset, tsfm_only=True)
        assert isinstance(models, list)
        # Should have TSFM models (chronos, timesfm, moirai)
        assert len(models) >= 3

    def test_make_plan_no_tsfm_raises(self, sample_df, config):
        """make_plan with no available TSFMs raises ENoTSFM."""
        from tsagentkit.core.errors import ENoTSFM

        dataset = TSDataset.from_dataframe(sample_df, config)
        # Mock empty registry by patching list_available
        import tsagentkit.pipeline as pipeline_module
        original_list_available = pipeline_module.list_available

        def mock_list_available(tsfm_only=True):
            return []

        pipeline_module.list_available = mock_list_available

        try:
            with pytest.raises(ENoTSFM):
                make_plan(dataset, tsfm_only=True)
        finally:
            pipeline_module.list_available = original_list_available


class TestEnsemble:
    """Test ensemble function."""

    def test_single_prediction(self):
        """Single prediction returns as-is."""
        pred = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        result = ensemble([pred], "median")
        assert len(result) == 3
        assert list(result["yhat"]) == [10.0, 20.0, 30.0]

    def test_median_ensemble(self):
        """Median ensemble computes correctly."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [30.0, 20.0, 10.0],
        })
        pred3 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [20.0, 20.0, 20.0],
        })
        result = ensemble([pred1, pred2, pred3], "median")
        assert list(result["yhat"]) == [20.0, 20.0, 20.0]

    def test_mean_ensemble(self):
        """Mean ensemble computes correctly."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [30.0, 20.0, 10.0],
        })
        result = ensemble([pred1, pred2], "mean")
        assert list(result["yhat"]) == [20.0, 20.0, 20.0]

    def test_empty_predictions_raises(self):
        """Empty predictions list raises EInsufficient."""
        with pytest.raises(EInsufficient, match="No predictions"):
            ensemble([], "median")

    def test_unknown_method_raises(self):
        """Unknown ensemble method raises ValueError."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [20.0, 30.0, 40.0],
        })
        # Need multiple predictions to reach the method validation
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            ensemble([pred1, pred2], "unknown")


class TestForecast:
    """Test forecast function (integration test)."""

    @pytest.mark.skip(reason="TSFM models require external dependencies not available in test environment")
    def test_forecast_basic(self, sample_df):
        """Basic forecast works."""
        # Use a simple dataset with enough data
        df = pd.DataFrame({
            "unique_id": ["A"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50),
            "y": range(50),
        })
        result = forecast(df, h=7)
        assert result is not None
        assert hasattr(result, "df")
        assert len(result.df) == 7  # One series, 7 periods

    @pytest.mark.skip(reason="TSFM models require external dependencies not available in test environment")
    def test_run_forecast_with_config(self, sample_df, config):
        """run_forecast with config works."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50),
            "y": range(50),
        })
        result = run_forecast(df, config)
        assert result is not None
        assert hasattr(result, "df")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
