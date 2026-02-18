"""Tests for consolidated pipeline.

Tests the new consolidated pipeline functions in pipeline.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.core.dataset import CovariateSet
from tsagentkit.core.errors import EContract, EInsufficient
from tsagentkit.models.ensemble import ensemble_with_quantiles as ensemble
from tsagentkit.pipeline import (
    build_dataset,
    forecast,
    make_plan,
    predict_all,
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

    def test_custom_columns_rejected(self, config):
        """Custom column names are rejected by fixed contract."""
        df = pd.DataFrame({
            "series_id": ["A"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "value": range(10),
        })
        with pytest.raises(EContract, match="Missing required columns"):
            validate(df, config)

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

    def test_make_plan_returns_models(self):
        """make_plan returns list of models."""
        models = make_plan(tsfm_only=True)
        assert isinstance(models, list)
        # Should have TSFM models (chronos, timesfm, moirai)
        assert len(models) >= 3

    def test_make_plan_no_tsfm_raises(self):
        """make_plan with no available TSFMs raises ENoTSFM."""
        from tsagentkit.core.errors import ENoTSFM

        # Mock empty registry by patching list_models
        import tsagentkit.pipeline as pipeline_module
        original_list_models = pipeline_module.list_models

        def mock_list_models(tsfm_only=True, available_only=False):
            return []

        pipeline_module.list_models = mock_list_models

        try:
            with pytest.raises(ENoTSFM):
                make_plan(tsfm_only=True)
        finally:
            pipeline_module.list_models = original_list_models

    def test_make_plan_uses_registry_not_availability_filter(self):
        """make_plan should not use dependency availability filtering for TSFMs."""
        import tsagentkit.pipeline as pipeline_module

        captured = {"available_only": None}
        original_list_models = pipeline_module.list_models

        def mock_list_models(tsfm_only=True, available_only=False):
            captured["available_only"] = available_only
            return ["chronos"]

        pipeline_module.list_models = mock_list_models
        try:
            models = make_plan(tsfm_only=True)
            assert len(models) == 1
            assert models[0].name == "chronos"
            assert captured["available_only"] is False
        finally:
            pipeline_module.list_models = original_list_models


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


class TestPredictAllQuantilePlumbing:
    """Test quantile forwarding from pipeline to protocol layer."""

    def test_predict_all_passes_quantiles_to_protocol(self, sample_df, config):
        """predict_all forwards quantiles to protocol_predict."""
        import tsagentkit.pipeline as pipeline_module

        dataset = build_dataset(sample_df, config)

        model = object()
        artifact = object()
        captured: dict[str, object] = {}

        def mock_protocol_predict(spec, artifact, dataset, h, quantiles=None):
            del spec, artifact, dataset, h
            captured["quantiles"] = quantiles
            return pd.DataFrame({
                "unique_id": ["A"] * 7,
                "ds": pd.date_range("2024-02-01", periods=7),
                "yhat": [1.0] * 7,
            })

        original_protocol_predict = pipeline_module.protocol_predict
        pipeline_module.protocol_predict = mock_protocol_predict
        try:
            preds = predict_all(
                [model],
                [artifact],
                dataset,
                h=7,
                quantiles=(0.1, 0.5, 0.9),
            )
            assert len(preds) == 1
            assert captured["quantiles"] == (0.1, 0.5, 0.9)
        finally:
            pipeline_module.protocol_predict = original_protocol_predict


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
