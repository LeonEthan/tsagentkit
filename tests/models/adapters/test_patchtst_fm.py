"""Unit tests for PatchTST adapter.

These tests use mocking to avoid requiring the actual model download.
For real model tests, see tests/ci/test_real_tsfm_smoke_gate.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset


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


class TestPatchTSTFMAdapterFunctions:
    """Test module-level functions."""

    def test_unload_function(self):
        """Test unload() clears the cache."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        # Set some cached model
        patchtst_fm._loaded_model = MagicMock()

        # Unload
        patchtst_fm.unload()

        # Verify cache is cleared
        assert patchtst_fm._loaded_model is None

    def test_fit_calls_load(self, sample_dataset):
        """Test fit() function calls load()."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        with patch.object(patchtst_fm, "load") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            result = patchtst_fm.fit(sample_dataset)

            mock_load.assert_called_once()
            assert result is mock_model

    def test_load_uses_transformers_model_name(self):
        """Test load() uses transformers model class."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        patchtst_fm._loaded_model = None
        patchtst_fm._default_model_name = "ibm-research/patchtst-fm-r1"

        mock_model = MagicMock()
        with patch("transformers.PatchTSTForPrediction.from_pretrained", return_value=mock_model) as mock_from_pretrained:
            loaded = patchtst_fm.load("ibm-research/patchtst-fm-r1")

        assert loaded is mock_model
        assert mock_from_pretrained.call_args.args[0] == "ibm-research/patchtst-fm-r1"

    def test_predict_single_series(self, sample_df, sample_config):
        """Test predict() with single series."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        # Reset module cache
        patchtst_fm._loaded_model = None

        # Setup mock model with prediction outputs
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # Shape: (batch=1, horizon=7, quantiles=9)
        mock_outputs.prediction_outputs = np.array([
            [[1.0] * 9, [2.0] * 9, [3.0] * 9, [4.0] * 9, [5.0] * 9, [6.0] * 9, [7.0] * 9]
        ])
        mock_model.return_value = mock_outputs

        # Create dataset
        dataset = TSDataset.from_dataframe(sample_df, sample_config)

        # Generate forecast with patched torch
        with patch("tsagentkit.models.adapters.tsfm.patchtst_fm.torch") as mock_torch:
            mock_no_grad = MagicMock()
            mock_no_grad.__enter__ = MagicMock(return_value=None)
            mock_no_grad.__exit__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value = mock_no_grad

            forecast = patchtst_fm.predict(mock_model, dataset, h=7)

        # Verify forecast structure
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns
        assert all(forecast["unique_id"] == "A")
        assert "past_values" in mock_model.call_args.kwargs

    def test_predict_multi_series(self, sample_config):
        """Test predict() with multiple series."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        # Reset module cache
        patchtst_fm._loaded_model = None

        # Setup mock model with prediction outputs
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.prediction_outputs = np.array([
            [[1.0] * 9, [2.0] * 9, [3.0] * 9, [4.0] * 9, [5.0] * 9, [6.0] * 9, [7.0] * 9]
        ])
        mock_model.return_value = mock_outputs

        # Create multi-series data
        np.random.seed(42)
        multi_df = pd.DataFrame({
            "unique_id": ["A"] * 30 + ["B"] * 30,
            "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
            "y": np.random.randn(60).cumsum() + 100,
        })
        dataset = TSDataset.from_dataframe(multi_df, sample_config)

        # Generate forecast with patched torch
        with patch("tsagentkit.models.adapters.tsfm.patchtst_fm.torch") as mock_torch:
            mock_no_grad = MagicMock()
            mock_no_grad.__enter__ = MagicMock(return_value=None)
            mock_no_grad.__exit__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value = mock_no_grad

            forecast = patchtst_fm.predict(mock_model, dataset, h=7)

        # Verify forecast structure
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

    def test_predict_with_single_quantile(self, sample_df, sample_config):
        """Test predict() handles single quantile output."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        # Reset module cache
        patchtst_fm._loaded_model = None

        # Setup mock model with single quantile output
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # Shape: (batch=1, horizon=7, quantiles=1)
        mock_outputs.prediction_outputs = np.array([
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
        ])
        mock_model.return_value = mock_outputs

        # Create dataset
        dataset = TSDataset.from_dataframe(sample_df, sample_config)

        # Generate forecast with patched torch
        with patch("tsagentkit.models.adapters.tsfm.patchtst_fm.torch") as mock_torch:
            mock_no_grad = MagicMock()
            mock_no_grad.__enter__ = MagicMock(return_value=None)
            mock_no_grad.__exit__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value = mock_no_grad

            forecast = patchtst_fm.predict(mock_model, dataset, h=7)

        # Verify forecast structure
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert "yhat" in forecast.columns

    def test_predict_with_quantile_predictions_quantile_first(self, sample_df, sample_config):
        """Test predict() handles quantile_predictions in (batch, quantile, horizon) shape."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        patchtst_fm._loaded_model = None

        mock_model = MagicMock()
        mock_model.config.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        mock_outputs = MagicMock()
        # Shape: (batch=1, quantiles=9, horizon=7)
        mock_outputs.quantile_predictions = np.array(
            [
                [
                    [1, 2, 3, 4, 5, 6, 7],
                    [11, 12, 13, 14, 15, 16, 17],
                    [21, 22, 23, 24, 25, 26, 27],
                    [31, 32, 33, 34, 35, 36, 37],
                    [41, 42, 43, 44, 45, 46, 47],
                    [51, 52, 53, 54, 55, 56, 57],
                    [61, 62, 63, 64, 65, 66, 67],
                    [71, 72, 73, 74, 75, 76, 77],
                    [81, 82, 83, 84, 85, 86, 87],
                ]
            ]
        )
        mock_model.return_value = mock_outputs

        dataset = TSDataset.from_dataframe(sample_df, sample_config)
        with patch("tsagentkit.models.adapters.tsfm.patchtst_fm.torch") as mock_torch:
            mock_no_grad = MagicMock()
            mock_no_grad.__enter__ = MagicMock(return_value=None)
            mock_no_grad.__exit__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value = mock_no_grad
            forecast = patchtst_fm.predict(mock_model, dataset, h=7)

        assert forecast["yhat"].tolist() == [41, 42, 43, 44, 45, 46, 47]

    def test_predict_with_quantile_predictions_horizon_first(self, sample_df, sample_config):
        """Test predict() handles quantile_predictions in (batch, horizon, quantile) shape."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        patchtst_fm._loaded_model = None

        mock_model = MagicMock()
        mock_model.config.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        mock_outputs = MagicMock()
        # Shape: (batch=1, horizon=7, quantiles=9)
        mock_outputs.quantile_predictions = np.array(
            [
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [11, 12, 13, 14, 15, 16, 17, 18, 19],
                    [21, 22, 23, 24, 25, 26, 27, 28, 29],
                    [31, 32, 33, 34, 35, 36, 37, 38, 39],
                    [41, 42, 43, 44, 45, 46, 47, 48, 49],
                    [51, 52, 53, 54, 55, 56, 57, 58, 59],
                    [61, 62, 63, 64, 65, 66, 67, 68, 69],
                ]
            ]
        )
        mock_model.return_value = mock_outputs

        dataset = TSDataset.from_dataframe(sample_df, sample_config)
        with patch("tsagentkit.models.adapters.tsfm.patchtst_fm.torch") as mock_torch:
            mock_no_grad = MagicMock()
            mock_no_grad.__enter__ = MagicMock(return_value=None)
            mock_no_grad.__exit__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value = mock_no_grad
            forecast = patchtst_fm.predict(mock_model, dataset, h=7)

        assert forecast["yhat"].tolist() == [5, 15, 25, 35, 45, 55, 65]

    def test_predict_with_transformers_style_model(self, sample_df, sample_config):
        """Test predict() handles transformers-style PatchTST interface."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        class DummyTransformersPatchTST:
            def __init__(self):
                self.config = type("Config", (), {"quantiles": None})()

            def forward(self, past_values):
                del past_values
                return type(
                    "Output",
                    (),
                    {"prediction_outputs": np.array([[[1], [2], [3], [4], [5], [6], [7]]])},
                )()
            __call__ = forward

        dataset = TSDataset.from_dataframe(sample_df, sample_config)
        model = DummyTransformersPatchTST()
        forecast = patchtst_fm.predict(model, dataset, h=7)

        assert forecast["yhat"].tolist() == [1, 2, 3, 4, 5, 6, 7]


class TestPatchTSTFMAdapterClass:
    """Test backward-compatible adapter class."""

    def test_adapter_init(self):
        """Test adapter initialization."""
        from tsagentkit.models.adapters.tsfm.patchtst_fm import PatchTSTFMAdapter

        adapter = PatchTSTFMAdapter()
        assert adapter.model_name == "ibm-research/patchtst-fm-r1"

        adapter_custom = PatchTSTFMAdapter(model_name="custom-model")
        assert adapter_custom.model_name == "custom-model"

    @patch("tsagentkit.models.adapters.tsfm.patchtst_fm.load")
    def test_adapter_fit(self, mock_load, sample_dataset):
        """Test adapter fit method."""
        from tsagentkit.models.adapters.tsfm.patchtst_fm import PatchTSTFMAdapter

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        adapter = PatchTSTFMAdapter()
        artifact = adapter.fit(sample_dataset)

        mock_load.assert_called_once()
        assert artifact["model"] is mock_model
        assert artifact["model_name"] == "ibm-research/patchtst-fm-r1"
        assert artifact["adapter"] is adapter

    @patch("tsagentkit.models.adapters.tsfm.patchtst_fm.predict")
    @patch("tsagentkit.models.adapters.tsfm.patchtst_fm.load")
    def test_adapter_predict(self, mock_load, mock_predict, sample_dataset):
        """Test adapter predict method."""
        from tsagentkit.models.adapters.tsfm.patchtst_fm import PatchTSTFMAdapter

        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_forecast = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-01", periods=7, freq="D"),
            "yhat": [1.0] * 7,
        })
        mock_predict.return_value = mock_forecast

        adapter = PatchTSTFMAdapter()
        artifact = {"model": mock_model, "adapter": adapter}

        forecast = adapter.predict(sample_dataset, artifact, h=7)

        mock_predict.assert_called_once_with(mock_model, sample_dataset, 7)
        assert forecast is mock_forecast

    @patch("tsagentkit.models.adapters.tsfm.patchtst_fm.predict")
    @patch("tsagentkit.models.adapters.tsfm.patchtst_fm.load")
    def test_adapter_predict_reloads_model(self, mock_load, mock_predict, sample_dataset):
        """Test adapter predict reloads model if not in artifact."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm
        from tsagentkit.models.adapters.tsfm.patchtst_fm import PatchTSTFMAdapter

        # Clear cache
        patchtst_fm._loaded_model = None

        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_forecast = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-01", periods=7, freq="D"),
            "yhat": [1.0] * 7,
        })
        mock_predict.return_value = mock_forecast

        adapter = PatchTSTFMAdapter()
        artifact = {}  # Empty artifact

        forecast = adapter.predict(sample_dataset, artifact, h=7)

        mock_load.assert_called_once()
        mock_predict.assert_called_once_with(mock_model, sample_dataset, 7)
        assert forecast is mock_forecast
