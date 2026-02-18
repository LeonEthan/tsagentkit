"""Tests for Chronos2 adapter with covariates support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tsagentkit import CovariateSet, ForecastConfig, TSDataset


@pytest.fixture
def sample_df():
    """Create sample time series data."""
    np.random.seed(42)
    return pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50, freq="D"),
        "y": np.random.randn(50).cumsum() + 100,
    })


@pytest.fixture
def sample_config():
    """Create sample forecast config."""
    return ForecastConfig(h=7, freq="D")


@pytest.fixture
def sample_dataset(sample_df, sample_config):
    """Create sample dataset."""
    return TSDataset.from_dataframe(sample_df, sample_config)


@pytest.fixture
def multi_series_df():
    """Create multi-series DataFrame."""
    np.random.seed(42)
    dfs = []
    for uid in ["A", "B"]:
        df = pd.DataFrame({
            "unique_id": [uid] * 40,
            "ds": pd.date_range("2024-01-01", periods=40, freq="D"),
            "y": np.random.randn(40).cumsum() + 100,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def future_covariates_df():
    """Create future covariates for horizon 7."""
    return pd.DataFrame({
        "unique_id": ["A"] * 7,
        "ds": pd.date_range("2024-02-20", periods=7, freq="D"),
        "promotion": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    })


@pytest.fixture
def future_covariates_multi_df():
    """Create future covariates for multiple series."""
    dfs = []
    for uid, promo_values in [
        ("A", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("B", [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
    ]:
        df = pd.DataFrame({
            "unique_id": [uid] * 7,
            "ds": pd.date_range("2024-02-10", periods=7, freq="D"),
            "promotion": promo_values,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class DummyChronosModel:
    """Simple stand-in for Chronos2Pipeline."""

    def __init__(self, return_constant: float | None = None):
        self.return_constant = return_constant
        self.calls: list[dict] = []

    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        feat_dynamic_real: torch.Tensor | None = None,
    ):
        """Mock predict method that records calls and returns dummy predictions."""
        batch_size = context.shape[0]

        self.calls.append({
            "context_shape": context.shape,
            "prediction_length": prediction_length,
            "has_covariates": feat_dynamic_real is not None,
            "covariates_shape": feat_dynamic_real.shape if feat_dynamic_real is not None else None,
        })

        # Return dummy predictions: (batch_size, n_samples, n_variates, prediction_length)
        if self.return_constant is not None:
            predictions = torch.full(
                (batch_size, 100, 1, prediction_length),
                self.return_constant,
                dtype=torch.float32,
            )
        else:
            predictions = torch.randn(batch_size, 100, 1, prediction_length)

        # Wrap in a tensor-like object that supports indexing and median
        return DummyPredictionTensor(predictions)


class DummyPredictionTensor:
    """Wrapper to mimic Chronos prediction tensor behavior."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getitem__(self, idx: int):
        """Return predictions for a single series."""
        return self.tensor[idx]

    def median(self, dim: int = 0):
        """Compute median along dimension."""
        result = self.tensor.median(dim=dim)
        return type("MedianResult", (), {"values": result.values})()


class TestChronosPredictWithoutCovariates:
    """Test predict() without covariates (baseline behavior)."""

    def test_predict_single_series_no_covariates(self, sample_dataset):
        """Test baseline predict without covariates."""
        from tsagentkit.models.adapters.tsfm import chronos

        model = DummyChronosModel(return_constant=50.0)
        forecast = chronos.predict(model, sample_dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert {"unique_id", "ds", "yhat"}.issubset(forecast.columns)
        assert all(forecast["yhat"] == 50.0)

        # Verify no covariates were passed
        assert len(model.calls) == 1
        assert model.calls[0]["has_covariates"] is False
        assert model.calls[0]["covariates_shape"] is None

    def test_predict_multi_series_no_covariates(self, multi_series_df, sample_config):
        """Test predict with multiple series without covariates."""
        from tsagentkit.models.adapters.tsfm import chronos

        dataset = TSDataset.from_dataframe(multi_series_df, sample_config)
        model = DummyChronosModel(return_constant=100.0)

        forecast = chronos.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

    def test_predict_outputs_requested_quantiles(self, sample_dataset):
        """Requested quantiles are exposed as q* columns."""
        from tsagentkit.models.adapters.tsfm import chronos

        model = DummyChronosModel(return_constant=42.0)
        forecast = chronos.predict(model, sample_dataset, h=7, quantiles=(0.1, 0.5, 0.9))

        assert {"q0.1", "q0.5", "q0.9"}.issubset(forecast.columns)
        assert np.allclose(forecast["q0.1"].values, 42.0)
        assert np.allclose(forecast["q0.5"].values, 42.0)
        assert np.allclose(forecast["q0.9"].values, 42.0)


class TestChronosPredictWithFutureCovariates:
    """Test predict() with future covariates (feat_dynamic_real)."""

    def test_predict_single_series_with_future_covariates(
        self, sample_df, sample_config, future_covariates_df
    ):
        """Test predict with future covariates."""
        from tsagentkit.models.adapters.tsfm import chronos

        covariates = CovariateSet(future=future_covariates_df)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = DummyChronosModel(return_constant=75.0)
        forecast = chronos.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert {"unique_id", "ds", "yhat"}.issubset(forecast.columns)

        # Verify covariates were passed
        assert len(model.calls) == 1
        assert model.calls[0]["has_covariates"] is True
        # Shape: (batch_size, n_features, context_length + h)
        assert model.calls[0]["covariates_shape"][0] == 1  # batch_size
        assert model.calls[0]["covariates_shape"][1] == 1  # n_features (promotion)
        assert model.calls[0]["covariates_shape"][2] == 50 + 7  # context + prediction

    def test_predict_multi_series_with_future_covariates(
        self, multi_series_df, sample_config, future_covariates_multi_df
    ):
        """Test predict with multiple series and covariates."""
        from tsagentkit.models.adapters.tsfm import chronos

        covariates = CovariateSet(future=future_covariates_multi_df)
        dataset = TSDataset.from_dataframe(multi_series_df, sample_config, covariates=covariates)

        model = DummyChronosModel(return_constant=100.0)
        forecast = chronos.predict(model, dataset, h=7, batch_size=2)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

        # Verify covariates were passed with batch
        assert len(model.calls) == 1
        assert model.calls[0]["has_covariates"] is True
        assert model.calls[0]["covariates_shape"][0] == 2  # batch_size

    def test_covariate_shape_alignment(self, sample_df, sample_config):
        """Test that covariate shapes align with variable-length series."""
        from tsagentkit.models.adapters.tsfm import chronos

        # Create covariates with multiple features
        future_cov = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-02-20", periods=7, freq="D"),
            "promotion": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "price": [10.0, 10.0, 9.0, 9.0, 10.0, 10.0, 10.0],
        })

        covariates = CovariateSet(future=future_cov)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = DummyChronosModel()
        chronos.predict(model, dataset, h=7)

        # Verify covariates have correct number of features
        assert model.calls[0]["covariates_shape"][1] == 2  # 2 features

    def test_missing_covariates_for_series(self, multi_series_df, sample_config):
        """Test behavior when some series have no covariates."""
        from tsagentkit.models.adapters.tsfm import chronos

        # Only provide covariates for series A
        future_cov = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-02-10", periods=7, freq="D"),
            "feature": [1.0] * 7,
        })

        covariates = CovariateSet(future=future_cov)
        dataset = TSDataset.from_dataframe(multi_series_df, sample_config, covariates=covariates)

        model = DummyChronosModel()
        forecast = chronos.predict(model, dataset, h=7)

        # Should still produce forecasts for both series
        assert len(forecast) == 14
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

        # Covariates should still be passed (zeros for missing series)
        assert model.calls[0]["has_covariates"] is True


class TestChronosCovariateBatching:
    """Test covariate handling with different batch sizes."""

    def test_batching_with_variable_length_series(self, sample_config):
        """Test that covariates handle variable-length series correctly."""
        from tsagentkit.models.adapters.tsfm import chronos

        # Create series with different lengths
        dfs = []
        for uid, length in [("A", 50), ("B", 30)]:
            df = pd.DataFrame({
                "unique_id": [uid] * length,
                "ds": pd.date_range("2024-01-01", periods=length, freq="D"),
                "y": range(length),
            })
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # Create covariates for both series
        cov_dfs = []
        for uid in ["A", "B"]:
            cov_df = pd.DataFrame({
                "unique_id": [uid] * 7,
                "ds": pd.date_range("2024-02-20" if uid == "A" else "2022-02-01", periods=7, freq="D"),
                "feature": [1.0] * 7,
            })
            cov_dfs.append(cov_df)
        covariates = CovariateSet(future=pd.concat(cov_dfs, ignore_index=True))

        dataset = TSDataset.from_dataframe(df, sample_config, covariates=covariates)

        model = DummyChronosModel()
        forecast = chronos.predict(model, dataset, h=7)

        assert len(forecast) == 14
        # Verify batch was processed together (both series in one batch)
        assert model.calls[0]["context_shape"][0] == 2
        assert model.calls[0]["covariates_shape"][0] == 2


class TestChronosEmptyCovariates:
    """Test behavior with empty or None covariates."""

    def test_empty_covariate_set(self, sample_df, sample_config):
        """Test that empty covariate set doesn't affect prediction."""
        from tsagentkit.models.adapters.tsfm import chronos

        covariates = CovariateSet()
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = DummyChronosModel()
        forecast = chronos.predict(model, dataset, h=7)

        assert len(forecast) == 7
        assert model.calls[0]["has_covariates"] is False

    def test_none_covariates(self, sample_dataset):
        """Test that None covariates doesn't affect prediction."""
        from tsagentkit.models.adapters.tsfm import chronos

        model = DummyChronosModel()
        forecast = chronos.predict(model, sample_dataset, h=7)

        assert len(forecast) == 7
        assert model.calls[0]["has_covariates"] is False

    def test_only_static_covariates(self, sample_df, sample_config):
        """Test that static covariates alone don't trigger feat_dynamic_real."""
        from tsagentkit.models.adapters.tsfm import chronos

        static_cov = pd.DataFrame({
            "unique_id": ["A"],
            "category": ["X"],
        })
        covariates = CovariateSet(static=static_cov)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = DummyChronosModel()
        forecast = chronos.predict(model, dataset, h=7)

        assert len(forecast) == 7
        # Static covariates should not enable feat_dynamic_real
        assert model.calls[0]["has_covariates"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
