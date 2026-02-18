"""Tests for quantile forecasting functionality.

Tests quantile configuration and ensemble aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.models import ensemble, ensemble_with_quantiles, fit, predict, get_spec


@pytest.fixture
def sample_df():
    """Create sample DataFrame with some noise."""
    np.random.seed(42)
    return pd.DataFrame({
        "unique_id": ["A"] * 100,
        "ds": pd.date_range("2024-01-01", periods=100),
        "y": np.sin(np.linspace(0, 8 * np.pi, 100)) * 10 + 20 + np.random.randn(100) * 2,
    })


@pytest.fixture
def multi_series_df():
    """Create multi-series DataFrame."""
    np.random.seed(42)
    dfs = []
    for uid in ["A", "B"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "y": np.sin(np.linspace(0, 8 * np.pi, 100)) * 10 + 20 + np.random.randn(100) * 2,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestQuantileConfiguration:
    """Test quantile configuration."""

    def test_default_quantiles(self):
        """Default quantiles are (0.1, 0.5, 0.9)."""
        config = ForecastConfig(h=7, freq="D")
        assert config.quantiles == (0.1, 0.5, 0.9)

    def test_custom_quantiles(self):
        """Can set custom quantiles."""
        config = ForecastConfig(h=7, freq="D", quantiles=(0.05, 0.25, 0.5, 0.75, 0.95))
        assert config.quantiles == (0.05, 0.25, 0.5, 0.75, 0.95)

    def test_single_quantile(self):
        """Can set single quantile."""
        config = ForecastConfig(h=7, freq="D", quantiles=(0.5,))
        assert config.quantiles == (0.5,)


class TestEnsembleAggregation:
    """Test ensemble aggregation methods."""

    def test_ensemble_median(self, sample_df):
        """Median ensemble of multiple models."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        # Get predictions from multiple models
        specs = [get_spec("naive"), get_spec("seasonal_naive")]
        predictions = []
        for spec in specs:
            artifact = fit(spec, dataset)
            pred = predict(spec, artifact, dataset, h=7)
            predictions.append(pred)

        result = ensemble(predictions, method="median")
        assert len(result) == 7
        assert "yhat" in result.columns

    def test_ensemble_mean(self, sample_df):
        """Mean ensemble of multiple models."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        specs = [get_spec("naive"), get_spec("seasonal_naive")]
        predictions = []
        for spec in specs:
            artifact = fit(spec, dataset)
            pred = predict(spec, artifact, dataset, h=7)
            predictions.append(pred)

        result = ensemble(predictions, method="mean")
        assert len(result) == 7
        assert "yhat" in result.columns

    def test_ensemble_single_prediction(self, sample_df):
        """Ensemble with single prediction returns that prediction."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        pred = predict(spec, artifact, dataset, h=7)

        result = ensemble([pred])
        pd.testing.assert_frame_equal(result, pred)


class TestEnsembleWithQuantiles:
    """Test ensemble with quantile columns."""

    def test_ensemble_with_manual_quantiles(self, sample_df):
        """Ensemble predictions that have quantile columns added manually."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        # Create predictions with manual quantile columns
        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        pred = predict(spec, artifact, dataset, h=7)

        # Simulate quantile columns (e.g., from different model runs)
        pred1 = pred.copy()
        pred1["q0.1"] = pred1["yhat"] * 0.9
        pred1["q0.9"] = pred1["yhat"] * 1.1

        pred2 = pred.copy()
        pred2["q0.1"] = pred2["yhat"] * 0.85
        pred2["q0.9"] = pred2["yhat"] * 1.15

        result = ensemble_with_quantiles([pred1, pred2], quantiles=[0.1, 0.9])

        assert "q0.1" in result.columns
        assert "q0.9" in result.columns

    def test_ensemble_quantile_uses_available_models_only(self, sample_df):
        """When some models miss quantiles, aggregate from available quantile columns only."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        pred = predict(spec, artifact, dataset, h=7)

        # First prediction has quantiles, second doesn't
        pred1 = pred.copy()
        pred1["q0.5"] = pred1["yhat"]

        pred2 = pred.copy()  # No quantile column

        result = ensemble_with_quantiles([pred1, pred2], quantiles=[0.5], quantile_mode="best_effort")

        assert "q0.5" in result.columns
        # Only pred1 contributes q0.5
        expected = pred1["q0.5"].values
        np.testing.assert_array_equal(result["q0.5"].values, expected)

    def test_ensemble_quantile_missing_best_effort_skips_column(self, sample_df):
        """best_effort mode skips requested quantiles if no model provides them."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        pred = predict(spec, artifact, dataset, h=7)

        result = ensemble_with_quantiles([pred, pred.copy()], quantiles=[0.5], quantile_mode="best_effort")
        assert "q0.5" not in result.columns

    def test_ensemble_quantile_missing_strict_raises(self, sample_df):
        """strict mode raises if requested quantiles are unavailable."""
        from tsagentkit.core.errors import EInsufficient

        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        spec = get_spec("naive")
        artifact = fit(spec, dataset)
        pred = predict(spec, artifact, dataset, h=7)

        with pytest.raises(EInsufficient, match="Requested quantile 'q0.5'"):
            ensemble_with_quantiles([pred, pred.copy()], quantiles=[0.5], quantile_mode="strict")


class TestEnsembleMultiSeries:
    """Test ensemble with multiple series."""

    def test_ensemble_multi_series(self, multi_series_df):
        """Ensemble works with multiple series."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(multi_series_df, config)

        specs = [get_spec("naive"), get_spec("seasonal_naive")]
        predictions = []
        for spec in specs:
            artifact = fit(spec, dataset)
            pred = predict(spec, artifact, dataset, h=7)
            predictions.append(pred)

        result = ensemble(predictions)

        assert len(result) == 14  # 2 series * 7 horizon
        assert set(result["unique_id"].unique()) == {"A", "B"}


class TestEnsembleEdgeCases:
    """Test ensemble edge cases."""

    def test_ensemble_empty_raises(self):
        """Empty predictions list raises EInsufficient."""
        from tsagentkit.core.errors import EInsufficient

        with pytest.raises(EInsufficient, match="No predictions"):
            ensemble([])

    def test_ensemble_unknown_method(self, sample_df):
        """Unknown ensemble method raises ValueError."""
        config = ForecastConfig(h=7, freq="D")
        dataset = TSDataset.from_dataframe(sample_df, config)

        # Need multiple predictions to trigger method validation
        specs = [get_spec("naive"), get_spec("seasonal_naive")]
        predictions = []
        for spec in specs:
            artifact = fit(spec, dataset)
            pred = predict(spec, artifact, dataset, h=7)
            predictions.append(pred)

        with pytest.raises(ValueError, match="Unknown ensemble method"):
            ensemble(predictions, method="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
