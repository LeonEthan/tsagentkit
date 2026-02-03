"""Tests for PRD-aligned router."""

import pandas as pd

from tsagentkit import TaskSpec
from tsagentkit.router import get_model_for_series, make_plan
from tsagentkit.series import TSDataset


def _make_dataset(values: list[float]) -> TSDataset:
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * len(values),
            "ds": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "y": values,
        }
    )
    spec = TaskSpec(h=7, freq="D")
    return TSDataset.from_dataframe(df, spec)


def test_make_plan_regular_series() -> None:
    dataset = _make_dataset([1.0] * 30)
    plan = make_plan(dataset, dataset.task_spec)
    assert plan.candidate_models[0] in {"HistoricAverage", "tsfm-chronos", "tsfm-moirai", "tsfm-timesfm"}


def test_make_plan_intermittent_series() -> None:
    dataset = _make_dataset([0.0] * 10 + [1.0] + [0.0] * 14 + [10.0] + [0.0] * 14 + [2.0])
    plan = make_plan(dataset, dataset.task_spec, use_tsfm=False)
    assert plan.candidate_models[0] == "Croston"


def test_get_model_for_series_short_history() -> None:
    dataset = _make_dataset([1.0] * 5)
    model = get_model_for_series("A", dataset, dataset.task_spec)
    assert model == "HistoricAverage"
