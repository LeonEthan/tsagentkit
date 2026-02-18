"""Unit tests for PatchTST-FM adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

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


class DummyPatchTSTFMModel:
    """Simple stand-in for PatchTSTFMForPrediction."""

    def __init__(
        self,
        context_length: int = 64,
        quantile_levels: list[float] | None = None,
        nan_output: bool = False,
    ):
        self.config = type(
            "Config",
            (),
            {
                "context_length": context_length,
                "quantile_levels": quantile_levels or [0.1, 0.5, 0.9],
            },
        )()
        self.device = torch.device("cpu")
        self.nan_output = nan_output
        self.calls: list[dict[str, object]] = []

    def eval(self):
        return self

    def __call__(
        self,
        *,
        inputs,
        prediction_length: int,
        quantile_levels: list[float] | None = None,
        return_loss: bool = False,
    ):
        del return_loss
        levels = quantile_levels or self.config.quantile_levels
        self.calls.append({
            "inputs": inputs,
            "prediction_length": prediction_length,
            "quantile_levels": levels,
        })

        if self.nan_output:
            predictions = np.full((1, len(levels), prediction_length), np.nan, dtype=np.float32)
        else:
            predictions = np.zeros((1, len(levels), prediction_length), dtype=np.float32)
            for i in range(len(levels)):
                predictions[0, i, :] = np.arange(1, prediction_length + 1, dtype=np.float32) + i * 10

        return type("Output", (), {"quantile_predictions": predictions})()


class TestPatchTSTFMAdapterFunctions:
    """Test module-level functions."""

    def test_unload_function(self):
        """Test unload() accepts model reference and is best-effort."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        model = MagicMock()
        patchtst_fm.unload(model)

    def test_fit_calls_load(self, sample_dataset):
        """Test fit() function calls load()."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        with patch.object(patchtst_fm, "load") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            result = patchtst_fm.fit(sample_dataset)

        mock_load.assert_called_once()
        assert result is mock_model

    def test_load_uses_patchtst_fm_model_name(self):
        """Test load() uses PatchTSTFMForPrediction class."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        mock_model = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        with (
            patch.object(patchtst_fm, "_get_patchtst_fm_class", return_value=mock_model_cls),
            patch.object(patchtst_fm, "_resolve_device_map", return_value="cpu"),
        ):
            loaded = patchtst_fm.load("ibm-research/patchtst-fm-r1")

        assert loaded is mock_model
        assert mock_model_cls.from_pretrained.call_args.args[0] == "ibm-research/patchtst-fm-r1"
        assert mock_model_cls.from_pretrained.call_args.kwargs["device_map"] == "cpu"

    def test_predict_single_series_uses_patchtst_fm_signature(self, sample_dataset):
        """Test predict() uses inputs/prediction_length/quantile_levels signature."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        model = DummyPatchTSTFMModel(context_length=64, quantile_levels=[0.1, 0.5, 0.9])
        forecast = patchtst_fm.predict(model, sample_dataset, h=7)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 7
        assert {"unique_id", "ds", "yhat"}.issubset(forecast.columns)
        assert forecast["yhat"].tolist() == [1, 2, 3, 4, 5, 6, 7]

        call = model.calls[0]
        assert call["prediction_length"] == 7
        assert call["quantile_levels"] == [0.5]
        assert isinstance(call["inputs"], list)
        context_tensor = call["inputs"][0]
        assert isinstance(context_tensor, torch.Tensor)
        assert context_tensor.shape[0] == 64

    def test_predict_outputs_requested_quantiles(self, sample_dataset):
        """predict() returns q* columns when quantiles are requested."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        model = DummyPatchTSTFMModel(context_length=64, quantile_levels=[0.1, 0.5, 0.9])
        forecast = patchtst_fm.predict(model, sample_dataset, h=7, quantiles=(0.1, 0.5, 0.9))

        assert {"q0.1", "q0.5", "q0.9"}.issubset(forecast.columns)
        assert forecast["q0.1"].tolist() == [1, 2, 3, 4, 5, 6, 7]
        assert forecast["q0.5"].tolist() == [11, 12, 13, 14, 15, 16, 17]
        assert forecast["q0.9"].tolist() == [21, 22, 23, 24, 25, 26, 27]

        call = model.calls[0]
        assert call["quantile_levels"] == [0.1, 0.5, 0.9]

    def test_predict_multi_series(self, sample_config):
        """Test predict() with multiple series."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        np.random.seed(42)
        multi_df = pd.DataFrame({
            "unique_id": ["A"] * 30 + ["B"] * 30,
            "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
            "y": np.random.randn(60).cumsum() + 100,
        })
        dataset = TSDataset.from_dataframe(multi_df, sample_config)
        model = DummyPatchTSTFMModel(context_length=16)

        forecast = patchtst_fm.predict(model, dataset, h=7)
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14  # 2 series * 7 horizon
        assert set(forecast["unique_id"].unique()) == {"A", "B"}

    def test_predict_nan_outputs_fallback_to_last_value(self, sample_dataset):
        """Test predict() replaces all-NaN forecast with last observed value."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        model = DummyPatchTSTFMModel(context_length=64, nan_output=True)
        forecast = patchtst_fm.predict(model, sample_dataset, h=7)

        expected = float(sample_dataset.df["y"].iloc[-1])
        assert forecast["yhat"].tolist() == pytest.approx([expected] * 7, rel=1e-6, abs=1e-6)

    def test_extract_forecast_values_horizon_first(self):
        """Test extraction supports horizon-first quantile outputs."""
        from tsagentkit.models.adapters.tsfm import patchtst_fm

        outputs = type(
            "Output",
            (),
            {
                "quantile_predictions": np.array(
                    [
                        [
                            [1, 2, 3],
                            [11, 12, 13],
                            [21, 22, 23],
                            [31, 32, 33],
                            [41, 42, 43],
                            [51, 52, 53],
                            [61, 62, 63],
                        ]
                    ]
                )
            },
        )()

        values = patchtst_fm._extract_forecast_values(
            outputs,
            h=7,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        assert values.tolist() == [2, 12, 22, 32, 42, 52, 62]
