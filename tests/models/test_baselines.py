from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.models.baselines import (
    fit_baseline,
    is_baseline_model,
    predict_baseline,
)
from tsagentkit.series import TSDataset


def test_baseline_model_aliases() -> None:
    """Test that baseline model aliases are recognized."""
    assert is_baseline_model("ETS")
    assert is_baseline_model("AutoETS")
    assert is_baseline_model("MovingAverage")
    assert is_baseline_model("WindowAverage")
    assert is_baseline_model("SeasonalNaive")
    assert is_baseline_model("HistoricAverage")
    assert is_baseline_model("Naive")
    assert is_baseline_model("Theta")
    assert is_baseline_model("Croston")


def test_ets_model_fit_predict() -> None:
    """Test ETS (AutoETS) model fitting and prediction."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": np.cumsum(np.random.randn(len(dates))) + 100,
    })

    spec = TaskSpec(horizon=7, freq="D", season_length=7)
    dataset = TSDataset.from_dataframe(df, spec)

    # Fit ETS model
    artifact = fit_baseline("ETS", dataset, {"season_length": 7})
    assert artifact.model_name == "ETS"
    assert artifact.model is not None

    # Predict
    forecast = predict_baseline(artifact, dataset, 7)
    assert len(forecast) == 7
    assert "unique_id" in forecast.columns
    assert "ds" in forecast.columns
    assert "yhat" in forecast.columns


def test_ets_with_quantiles() -> None:
    """Test ETS model with quantile forecasts."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": np.cumsum(np.random.randn(len(dates))) + 100,
    })

    spec = TaskSpec(horizon=7, freq="D", season_length=7)
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = fit_baseline("ETS", dataset, {"season_length": 7})
    forecast = predict_baseline(artifact, dataset, 7, quantiles=[0.1, 0.5, 0.9])

    # Quantile columns are normalized (e.g., q0.1 not q0.10)
    assert "q0.1" in forecast.columns
    assert "q0.5" in forecast.columns
    assert "q0.9" in forecast.columns


def test_seasonal_naive_model() -> None:
    """Test SeasonalNaive model."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": list(range(len(dates))),
    })

    spec = TaskSpec(horizon=7, freq="D", season_length=7)
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = fit_baseline("SeasonalNaive", dataset, {"season_length": 7})
    assert artifact.model_name == "SeasonalNaive"

    forecast = predict_baseline(artifact, dataset, 7)
    assert len(forecast) == 7


def test_naive_model() -> None:
    """Test Naive model."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": list(range(len(dates))),
    })

    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = fit_baseline("Naive", dataset, {})
    assert artifact.model_name == "Naive"

    forecast = predict_baseline(artifact, dataset, 7)
    assert len(forecast) == 7


def test_historic_average_model() -> None:
    """Test HistoricAverage model."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": [100.0] * len(dates),
    })

    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = fit_baseline("HistoricAverage", dataset, {})
    assert artifact.model_name == "HistoricAverage"

    forecast = predict_baseline(artifact, dataset, 7)
    assert len(forecast) == 7
    # Historic average should predict the mean
    assert all(abs(forecast["yhat"] - 100.0) < 0.01)


def test_theta_model() -> None:
    """Test Theta model."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": np.cumsum(np.random.randn(len(dates))) + 100,
    })

    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = fit_baseline("Theta", dataset, {})
    assert artifact.model_name == "Theta"

    forecast = predict_baseline(artifact, dataset, 7)
    assert len(forecast) == 7


def test_croston_model() -> None:
    """Test Croston model for intermittent demand."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    # Create intermittent demand (many zeros)
    np.random.seed(42)
    y_values = []
    for _ in range(len(dates)):
        if np.random.random() < 0.6:
            y_values.append(0)
        else:
            y_values.append(np.random.poisson(5))

    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": y_values,
    })

    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    # Croston may not be available in all statsforecast versions
    if is_baseline_model("Croston"):
        artifact = fit_baseline("Croston", dataset, {})
        assert "Croston" in artifact.model_name or "croston" in artifact.model_name.lower()

        forecast = predict_baseline(artifact, dataset, 7)
        assert len(forecast) == 7


def test_multiple_series() -> None:
    """Test baseline models with multiple series."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates) + ["B"] * len(dates),
        "ds": list(dates) * 2,
        "y": list(range(len(dates))) + list(range(50, 50 + len(dates))),
    })

    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = fit_baseline("Naive", dataset, {})
    forecast = predict_baseline(artifact, dataset, 7)

    # Should have 7 forecasts per series = 14 total
    assert len(forecast) == 14
    assert set(forecast["unique_id"].unique()) == {"A", "B"}


def test_unknown_model_raises() -> None:
    """Test that unknown model raises ValueError."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "unique_id": ["A"] * len(dates),
        "ds": dates,
        "y": list(range(len(dates))),
    })

    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    with pytest.raises(ValueError, match="Unknown baseline model"):
        fit_baseline("UnknownModel", dataset, {})
