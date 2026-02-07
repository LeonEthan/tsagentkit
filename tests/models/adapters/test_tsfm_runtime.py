"""Real TSFM runtime smoke tests."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from tsagentkit.contracts import TaskSpec
from tsagentkit.models.adapters import AdapterConfig
from tsagentkit.series import TSDataset

if os.environ.get("TSFM_RUN_REAL") != "1":
    pytestmark = pytest.mark.skip(
        reason="Real TSFM runtime smoke tests require TSFM_RUN_REAL=1.",
    )


def _make_dataset(freq: str = "D", periods: int = 32) -> TSDataset:
    df = pd.DataFrame({
        "unique_id": ["A"] * periods + ["B"] * periods,
        "ds": list(pd.date_range("2020-01-01", periods=periods, freq=freq)) * 2,
        "y": list(range(periods)) + list(range(periods, 0, -1)),
    })
    spec = TaskSpec(h=4, freq=freq)
    return TSDataset.from_dataframe(df, spec)


@pytest.mark.tsfm
def test_chronos2_smoke() -> None:
    from tsagentkit.models.adapters.chronos import ChronosAdapter

    dataset = _make_dataset()
    config = AdapterConfig(
        model_name="chronos",
        model_size="small",
        device="cpu",
        num_samples=8,
    )
    adapter = ChronosAdapter(config)
    result = adapter.predict(dataset, horizon=dataset.task_spec.horizon, quantiles=[0.1, 0.5, 0.9])

    assert not result.df.empty
    assert len(result.df) == dataset.n_series * dataset.task_spec.horizon
    assert "yhat" in result.df.columns


@pytest.mark.tsfm
def test_timesfm25_smoke() -> None:
    from tsagentkit.models.adapters.timesfm import TimesFMAdapter

    dataset = _make_dataset()
    config = AdapterConfig(
        model_name="timesfm",
        model_size="base",
        device="cpu",
    )
    adapter = TimesFMAdapter(config)
    result = adapter.predict(dataset, horizon=dataset.task_spec.horizon, quantiles=[0.1, 0.5, 0.9])

    assert not result.df.empty
    assert len(result.df) == dataset.n_series * dataset.task_spec.horizon
    assert "yhat" in result.df.columns


@pytest.mark.tsfm
def test_moirai11_smoke() -> None:
    from tsagentkit.models.adapters.moirai import MoiraiAdapter

    dataset = _make_dataset()
    config = AdapterConfig(
        model_name="moirai",
        model_size="small",
        device="cpu",
        num_samples=8,
    )
    adapter = MoiraiAdapter(config)
    result = adapter.predict(dataset, horizon=dataset.task_spec.horizon, quantiles=[0.1, 0.5, 0.9])

    assert not result.df.empty
    assert len(result.df) == dataset.n_series * dataset.task_spec.horizon
    assert "yhat" in result.df.columns
