"""Tests for pipeline stages.

Tests individual pipeline stage functions in isolation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.core.data import CovariateSet
from tsagentkit.core.errors import EContractViolation, EDataQuality, EModelFailed
from tsagentkit.pipeline.stages import (
    PipelineStage,
    _compute_ensemble,
    backtest_stage,
    build_dataset_stage,
    package_stage,
    qa_stage,
    validate_stage,
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
    return ForecastConfig(h=7, freq="D", tsfm_mode="disabled")


class TestValidateStage:
    """Test validate_stage function."""

    def test_valid_data_passes(self, sample_df, config):
        """Valid data passes validation."""
        result = validate_stage(sample_df, config)
        assert result is not None
        assert len(result) == 30

    def test_missing_column_raises(self, sample_df, config):
        """Missing column raises EContractViolation."""
        df_missing = sample_df.drop(columns=["y"])
        with pytest.raises(EContractViolation) as exc_info:
            validate_stage(df_missing, config)
        assert "Missing required columns" in str(exc_info.value)

    def test_empty_dataframe_raises(self, config):
        """Empty DataFrame raises EContractViolation."""
        empty_df = pd.DataFrame(columns=["unique_id", "ds", "y"])
        with pytest.raises(EContractViolation, match="empty"):
            validate_stage(empty_df, config)

    def test_null_in_unique_id_raises(self, sample_df, config):
        """Null in unique_id raises EContractViolation."""
        df_with_null = sample_df.copy()
        df_with_null.loc[0, "unique_id"] = None
        with pytest.raises(EContractViolation, match="null"):
            validate_stage(df_with_null, config)

    def test_null_in_ds_raises(self, sample_df, config):
        """Null in ds raises EContractViolation."""
        df_with_null = sample_df.copy()
        df_with_null.loc[0, "ds"] = None
        with pytest.raises(EContractViolation, match="null"):
            validate_stage(df_with_null, config)

    def test_null_in_y_raises(self, sample_df, config):
        """Null in y raises EContractViolation."""
        df_with_null = sample_df.copy()
        df_with_null.loc[0, "y"] = None
        with pytest.raises(EContractViolation, match="null"):
            validate_stage(df_with_null, config)

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
        result = validate_stage(df, custom_config)
        # Should be renamed to standard column names
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns

    def test_returns_copy(self, sample_df, config):
        """validate_stage returns a copy of the DataFrame."""
        result = validate_stage(sample_df, config)
        result["y"] = 0
        # Original should be unchanged
        assert sample_df["y"].iloc[0] == 0


class TestQAStage:
    """Test qa_stage function."""

    def test_valid_data_passes(self, sample_df, config):
        """Valid data passes QA."""
        result = qa_stage(sample_df, config)
        assert result is not None

    def test_duplicate_keys_raises(self, sample_df, config):
        """Duplicate (unique_id, ds) pairs raise EDataQuality."""
        df_with_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(EDataQuality) as exc_info:
            qa_stage(df_with_dup, config)
        assert "duplicate" in str(exc_info.value).lower()
        assert "drop_duplicates" in str(exc_info.value)

    def test_unsorted_series_raises(self, sample_df, config):
        """Unsorted series raises EDataQuality."""
        df_unsorted = sample_df.copy()
        df_unsorted["ds"] = df_unsorted["ds"].sample(frac=1).values
        with pytest.raises(EDataQuality) as exc_info:
            qa_stage(df_unsorted, config)
        assert "not sorted" in str(exc_info.value).lower()

    def test_short_series_warning_mode(self, sample_df, config):
        """Short series is allowed in non-strict mode."""
        # config.min_train_size is 56 by default, but we only have 30
        # This should just log a warning, not raise
        result = qa_stage(sample_df, config)
        assert result is not None

    def test_short_series_strict_mode(self):
        """Short series raises error in strict mode."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "y": range(10),
        })
        strict_config = ForecastConfig(h=3, freq="D", mode="strict", min_train_size=20)
        with pytest.raises(EDataQuality) as exc_info:
            qa_stage(df, strict_config)
        assert "strict mode" in str(exc_info.value).lower()


class TestBuildDatasetStage:
    """Test build_dataset_stage function."""

    def test_build_dataset(self, sample_df, config):
        """Build TSDataset from DataFrame."""
        dataset = build_dataset_stage(sample_df, config)
        assert isinstance(dataset, TSDataset)
        assert dataset.config == config

    def test_build_dataset_with_covariates(self, sample_df, config):
        """Build TSDataset with covariates."""
        cov = CovariateSet(static=pd.DataFrame({"unique_id": ["A"], "cat": ["X"]}))
        dataset = build_dataset_stage(sample_df, config, covariates=cov)
        assert dataset.covariates == cov


class TestBacktestStage:
    """Test backtest_stage function."""

    def test_backtest_skipped_when_zero_windows(self, sample_df, config):
        """Backtest skipped when n_backtest_windows is 0."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        plan = {"dummy": "plan"}
        zero_config = ForecastConfig(h=7, freq="D", n_backtest_windows=0)
        _, _, metrics = backtest_stage(dataset, plan, zero_config)
        assert metrics is None

    def test_backtest_skipped_in_quick_mode(self, sample_df, config):
        """Backtest skipped in quick mode."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        plan = {"dummy": "plan"}
        quick_config = ForecastConfig.quick(h=7, freq="D")
        _, _, metrics = backtest_stage(dataset, plan, quick_config)
        assert metrics is None


class TestComputeEnsemble:
    """Test _compute_ensemble function."""

    def test_single_prediction(self):
        """Single prediction returns as-is."""
        pred = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [10.0, 20.0, 30.0],
        })
        result = _compute_ensemble([pred], "median")
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
        result = _compute_ensemble([pred1, pred2, pred3], "median")
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
        result = _compute_ensemble([pred1, pred2], "mean")
        assert list(result["yhat"]) == [20.0, 20.0, 20.0]

    def test_ensemble_with_ensemble_count(self):
        """Ensemble includes count of models."""
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
        result = _compute_ensemble([pred1, pred2], "median")
        assert "_ensemble_count" in result.columns
        assert all(result["_ensemble_count"] == 2)

    def test_empty_predictions_raises(self):
        """Empty predictions list raises EModelFailed."""
        with pytest.raises(EModelFailed, match="No predictions"):
            _compute_ensemble([], "median")

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
            _compute_ensemble([pred1, pred2], "unknown")

    def test_quantile_ensemble_median(self):
        """Quantile columns are ensembled with median."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [20.0, 20.0, 20.0],
            "q0.1": [10.0, 10.0, 10.0],
            "q0.9": [30.0, 30.0, 30.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [30.0, 30.0, 30.0],
            "q0.1": [20.0, 20.0, 20.0],
            "q0.9": [40.0, 40.0, 40.0],
        })
        result = _compute_ensemble([pred1, pred2], "median", quantiles=[0.1, 0.5, 0.9])
        assert "q0.1" in result.columns
        assert "q0.9" in result.columns
        # Median of [10, 20] and [30, 40]
        assert list(result["q0.1"]) == [15.0, 15.0, 15.0]
        assert list(result["q0.9"]) == [35.0, 35.0, 35.0]

    def test_quantile_ensemble_with_fallback(self):
        """Quantile ensemble falls back to yhat if column missing."""
        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [20.0, 20.0, 20.0],
            "q0.1": [10.0, 10.0, 10.0],
        })
        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3),
            "yhat": [30.0, 30.0, 30.0],
            # No q0.1 column
        })
        result = _compute_ensemble([pred1, pred2], "median", quantiles=[0.1])
        # q0.1 should use yhat from pred2 as fallback (30.0)
        # Median of [10, 30]
        assert list(result["q0.1"]) == [20.0, 20.0, 20.0]


class TestPackageStage:
    """Test package_stage function."""

    def test_package_stage(self, sample_df, config):
        """Package results into RunResult."""
        from tsagentkit.core.results import ForecastResult, RunResult

        forecast_df = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-31", periods=7),
            "yhat": range(7),
        })
        forecast = ForecastResult(df=forecast_df, model_name="test", config=config)
        model_errors = [{"model": "failed_model", "error": "test error"}]

        result = package_stage(forecast, model_errors, config, 123.45)

        assert isinstance(result, RunResult)
        assert result.forecast == forecast
        assert result.duration_ms == 123.45
        assert result.model_used == "test"
        assert result.model_errors == model_errors


class TestPipelineStage:
    """Test PipelineStage dataclass."""

    def test_stage_creation(self):
        """Create pipeline stage."""
        def dummy_func():
            return "test"

        stage = PipelineStage(name="test_stage", run=dummy_func)
        assert stage.name == "test_stage"
        assert stage.run() == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
