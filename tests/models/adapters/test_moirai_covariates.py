"""Tests for Moirai2 adapter with covariates support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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
def past_covariates_df():
    """Create past covariates."""
    return pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50, freq="D"),
        "temperature": np.random.randn(50) * 10 + 20,
    })


class DummyForecastSample:
    """Simple stand-in for Forecast output."""

    def __init__(self, values: np.ndarray):
        self.values = values

    def quantile(self, q: float) -> np.ndarray:
        """Return median (or constant) forecast."""
        return self.values


class DummyMoiraiPredictor:
    """Simple stand-in for Moirai predictor."""

    def __init__(self, prediction_length: int, return_constant: float | None = None):
        self.prediction_length = prediction_length
        self.return_constant = return_constant
        self.predict_calls: list[dict] = []

    def predict(self, dataset):
        """Mock predict method that yields forecasts."""
        self.predict_calls.append({"dataset": dataset})

        # Get number of series from dataset
        n_series = len(dataset) if hasattr(dataset, "__len__") else 1

        for _ in range(n_series):
            if self.return_constant is not None:
                values = np.full(self.prediction_length, self.return_constant)
            else:
                values = np.random.randn(self.prediction_length) * 10 + 100
            yield DummyForecastSample(values)


class DummyMoirai2Forecast:
    """Simple stand-in for Moirai2Forecast class."""

    def __init__(
        self,
        module: object,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
    ):
        self.module = module
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.predictor_kwargs: dict | None = None

    def create_predictor(self, batch_size: int = 1):
        """Create a dummy predictor."""
        self.predictor_kwargs = {"batch_size": batch_size}
        return DummyMoiraiPredictor(
            prediction_length=self.prediction_length,
            return_constant=100.0,
        )


# Patch path for Moirai2Forecast (imported inside predict function)
MOIRAI_PATCH_PATH = "uni2ts.model.moirai2.Moirai2Forecast"


class TestMoiraiPredictWithoutCovariates:
    """Test predict() without covariates (baseline behavior)."""

    def test_predict_single_series_no_covariates(self, sample_dataset):
        """Test baseline predict without covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        model = {"module": MagicMock(), "model_name": "test-model"}

        with patch(MOIRAI_PATCH_PATH, DummyMoirai2Forecast):
            forecast = moirai.predict(model, sample_dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert {"unique_id", "ds", "yhat"}.issubset(forecast.columns)

    def test_predict_multi_series_no_covariates(self, multi_series_df, sample_config):
        """Test predict with multiple series without covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        dataset = TSDataset.from_dataframe(multi_series_df, sample_config)
        model = {"module": MagicMock(), "model_name": "test-model"}

        with patch(MOIRAI_PATCH_PATH, DummyMoirai2Forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

    def test_predict_sanitizes_non_finite_context(self, sample_df, sample_config):
        """Test that NaN/Inf targets are cleaned before creating PandasDataset."""
        from tsagentkit.models.adapters.tsfm import moirai

        df = sample_df.copy()
        df.loc[[0, 5, 10], "y"] = [np.nan, np.inf, -np.inf]
        dataset = TSDataset.from_dataframe(df, sample_config)
        model = {"module": MagicMock(), "model_name": "test-model"}

        captured_ts_dict: dict[str, pd.Series | pd.DataFrame] = {}

        class CapturedPandasDataset:
            def __init__(self, ts_dict, **kwargs):
                del kwargs
                captured_ts_dict.update(ts_dict)
                self._n = len(ts_dict)

            def __len__(self):
                return self._n

        with (
            patch(MOIRAI_PATCH_PATH, DummyMoirai2Forecast),
            patch("gluonts.dataset.pandas.PandasDataset", CapturedPandasDataset),
        ):
            forecast = moirai.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "A" in captured_ts_dict

        series = captured_ts_dict["A"]
        assert isinstance(series, pd.Series)
        assert np.isfinite(series.to_numpy(dtype=np.float32)).all()


class TestMoiraiPredictWithFutureCovariates:
    """Test predict() with future covariates (feat_dynamic_real)."""

    def test_predict_single_series_with_future_covariates(
        self, sample_df, sample_config, future_covariates_df
    ):
        """Test predict with future covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        covariates = CovariateSet(future=future_covariates_df)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7

        # Verify Moirai2Forecast was created with correct dimensions
        assert len(captured_forecast_args) == 1
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 1  # promotion feature
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 0

    def test_predict_multi_series_with_future_covariates(self, multi_series_df, sample_config):
        """Test predict with multiple series and future covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        # Create covariates for both series
        cov_dfs = []
        for uid in ["A", "B"]:
            df = pd.DataFrame({
                "unique_id": [uid] * 7,
                "ds": pd.date_range("2024-02-10" if uid == "A" else "2024-02-02", periods=7, freq="D"),
                "promotion": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "discount": [0.1, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
            })
            cov_dfs.append(df)
        future_cov = pd.concat(cov_dfs, ignore_index=True)

        covariates = CovariateSet(future=future_cov)
        dataset = TSDataset.from_dataframe(multi_series_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14

        # Verify correct number of features (promotion + discount)
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 2

    def test_future_covariate_dataframe_structure(self, sample_df, sample_config, future_covariates_df):
        """Test that PandasDataset receives correct structure with covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        covariates = CovariateSet(future=future_covariates_df)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_datasets = []

        def capture_pandas_dataset(ts_dict, **kwargs):
            captured_datasets.append(ts_dict)
            # Return a mock that can be iterated
            mock_iter = MagicMock()
            mock_iter.__iter__ = MagicMock(return_value=iter([]))
            mock_iter.__len__ = MagicMock(return_value=len(ts_dict))
            return mock_iter

        # PandasDataset is imported inside predict(), so patch it at gluonts
        with (
            patch(MOIRAI_PATCH_PATH, DummyMoirai2Forecast),
            patch("gluonts.dataset.pandas.PandasDataset", capture_pandas_dataset),
        ):
            moirai.predict(model, dataset, h=7)

        # Verify dataset structure if it was captured
        if captured_datasets:
            ts_dict = captured_datasets[0]
            assert "A" in ts_dict
            # Should be a DataFrame with target and covariate columns
            series_df = ts_dict["A"]
            assert isinstance(series_df, pd.DataFrame)
            assert "target" in series_df.columns
            assert any("feat_dynamic_real" in col for col in series_df.columns)


class TestMoiraiPredictWithPastCovariates:
    """Test predict() with past covariates (past_feat_dynamic_real)."""

    def test_predict_single_series_with_past_covariates(
        self, sample_df, sample_config, past_covariates_df
    ):
        """Test predict with past covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        covariates = CovariateSet(past=past_covariates_df)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7

        # Verify Moirai2Forecast was created with correct dimensions
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 0
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 1  # temperature feature


class TestMoiraiPredictWithBothCovariates:
    """Test predict() with both past and future covariates."""

    def test_predict_with_both_covariate_types(
        self, sample_df, sample_config, future_covariates_df, past_covariates_df
    ):
        """Test predict with both past and future covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        covariates = CovariateSet(future=future_covariates_df, past=past_covariates_df)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7

        # Verify both dimensions are set
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 1  # promotion
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 1  # temperature

    def test_multiple_features_both_types(self, sample_df, sample_config):
        """Test with multiple features for both past and future covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        # Create future covariates with multiple features
        future_cov = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-02-20", periods=7, freq="D"),
            "promotion": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "holiday": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        })

        # Create past covariates with multiple features
        past_cov = pd.DataFrame({
            "unique_id": ["A"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50, freq="D"),
            "temperature": np.random.randn(50) * 10 + 20,
            "rainfall": np.random.randn(50) * 5,
        })

        covariates = CovariateSet(future=future_cov, past=past_cov)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            moirai.predict(model, dataset, h=7)

        # Verify correct dimensions for multiple features
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 2  # promotion + holiday
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 2  # temperature + rainfall


class TestMoiraiCovariateBatching:
    """Test covariate handling with different batch sizes."""

    def test_batching_preserves_covariates(self, multi_series_df, sample_config):
        """Test that covariates are correctly assigned in batches."""
        from tsagentkit.models.adapters.tsfm import moirai

        # Create covariates for both series
        cov_dfs = []
        for uid in ["A", "B"]:
            df = pd.DataFrame({
                "unique_id": [uid] * 7,
                "ds": pd.date_range("2024-02-10" if uid == "A" else "2024-02-02", periods=7, freq="D"),
                "feature": [1.0] * 7,
            })
            cov_dfs.append(df)
        future_cov = pd.concat(cov_dfs, ignore_index=True)

        covariates = CovariateSet(future=future_cov)
        dataset = TSDataset.from_dataframe(multi_series_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}

        with patch(MOIRAI_PATCH_PATH, DummyMoirai2Forecast):
            forecast = moirai.predict(model, dataset, h=7, batch_size=2)

        # Should produce forecasts for both series
        assert len(forecast) == 14
        assert set(forecast["unique_id"].unique()) == {"A", "B"}


class TestMoiraiEmptyCovariates:
    """Test behavior with empty or None covariates."""

    def test_empty_covariate_set(self, sample_df, sample_config):
        """Test that empty covariate set doesn't affect prediction."""
        from tsagentkit.models.adapters.tsfm import moirai

        covariates = CovariateSet()
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert len(forecast) == 7
        # Both dimensions should be 0
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 0
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 0

    def test_none_covariates(self, sample_dataset):
        """Test that None covariates doesn't affect prediction."""
        from tsagentkit.models.adapters.tsfm import moirai

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, sample_dataset, h=7)

        assert len(forecast) == 7
        # Both dimensions should be 0
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 0
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 0

    def test_only_static_covariates(self, sample_df, sample_config):
        """Test that static covariates alone don't trigger dynamic covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        static_cov = pd.DataFrame({
            "unique_id": ["A"],
            "category": ["X"],
        })
        covariates = CovariateSet(static=static_cov)
        dataset = TSDataset.from_dataframe(sample_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}
        captured_forecast_args = []

        def capture_forecast(*args, **kwargs):
            captured_forecast_args.append(kwargs)
            return DummyMoirai2Forecast(*args, **kwargs)

        with patch(MOIRAI_PATCH_PATH, capture_forecast):
            forecast = moirai.predict(model, dataset, h=7)

        assert len(forecast) == 7
        # Static covariates should not enable dynamic covariates
        assert captured_forecast_args[0]["feat_dynamic_real_dim"] == 0
        assert captured_forecast_args[0]["past_feat_dynamic_real_dim"] == 0


class TestMoiraiMissingCovariates:
    """Test behavior when some series have missing covariates."""

    def test_partial_covariate_coverage(self, multi_series_df, sample_config):
        """Test when only some series have covariates."""
        from tsagentkit.models.adapters.tsfm import moirai

        # Only provide covariates for series A
        future_cov = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-02-10", periods=7, freq="D"),
            "feature": [1.0] * 7,
        })

        covariates = CovariateSet(future=future_cov)
        dataset = TSDataset.from_dataframe(multi_series_df, sample_config, covariates=covariates)

        model = {"module": MagicMock(), "model_name": "test-model"}

        with patch(MOIRAI_PATCH_PATH, DummyMoirai2Forecast):
            forecast = moirai.predict(model, dataset, h=7)

        # Should still produce forecasts for both series
        assert len(forecast) == 14
        assert set(forecast["unique_id"].unique()) == {"A", "B"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
