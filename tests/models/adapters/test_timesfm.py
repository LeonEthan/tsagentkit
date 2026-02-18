"""Unit tests for TimesFM adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsagentkit import ForecastConfig, TSDataset


class DummyTimesFMModel:
    """Simple stand-in for TimesFM model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.model = type(
            "DummyInnerModel",
            (),
            {"config": type("DummyConfig", (), {"quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})()},
        )()

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append({"horizon": horizon, "inputs": inputs})

        batch_size = len(inputs)
        point = np.zeros((batch_size, horizon), dtype=np.float32)
        quant = np.zeros((batch_size, horizon, 10), dtype=np.float32)

        for b in range(batch_size):
            point[b, :] = np.arange(1, horizon + 1, dtype=np.float32) + b
            for q_idx in range(10):
                quant[b, :, q_idx] = point[b, :] + q_idx * 10.0

        return point, quant


def _sample_dataset() -> TSDataset:
    df = pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50, freq="D"),
        "y": np.arange(50, dtype=np.float32),
    })
    config = ForecastConfig(h=7, freq="D")
    return TSDataset.from_dataframe(df, config)


def test_predict_without_quantiles_returns_point_forecast_only():
    """predict() returns yhat without q* columns by default."""
    from tsagentkit.models.adapters.tsfm import timesfm

    dataset = _sample_dataset()
    model = DummyTimesFMModel()

    forecast = timesfm.predict(model, dataset, h=7)

    assert list(forecast.columns) == ["unique_id", "ds", "yhat"]
    assert forecast["yhat"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    # Context is left-padded to TimesFM minimum context length.
    assert len(model.calls) == 1
    first_input = model.calls[0]["inputs"][0]
    assert isinstance(first_input, np.ndarray)
    assert first_input.shape[0] == 993


def test_predict_with_quantiles_returns_q_columns():
    """predict() returns q* columns for requested quantiles."""
    from tsagentkit.models.adapters.tsfm import timesfm

    dataset = _sample_dataset()
    model = DummyTimesFMModel()

    forecast = timesfm.predict(model, dataset, h=7, quantiles=(0.1, 0.5, 0.9))

    assert {"q0.1", "q0.5", "q0.9"}.issubset(forecast.columns)
    assert forecast["q0.1"].tolist() == [11, 12, 13, 14, 15, 16, 17]
    assert forecast["q0.5"].tolist() == [51, 52, 53, 54, 55, 56, 57]
    assert forecast["q0.9"].tolist() == [91, 92, 93, 94, 95, 96, 97]
