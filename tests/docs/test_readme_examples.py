"""Smoke tests for README-style examples."""

from __future__ import annotations

import pandas as pd

from tsagentkit import TaskSpec, make_plan, run_forecast
from tsagentkit.contracts import ModelArtifact
from tsagentkit.series import TSDataset


def _make_panel(n_points: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return pd.DataFrame(
        {
            "unique_id": ["A"] * n_points + ["B"] * n_points,
            "ds": list(dates) * 2,
            "y": list(range(n_points)) + list(range(n_points, 0, -1)),
        }
    )


def _fit_stub(dataset: TSDataset, plan) -> ModelArtifact:
    model_name = plan.candidate_models[0] if plan.candidate_models else "model"
    return ModelArtifact(model={"fitted": True}, model_name=model_name)


def _predict_stub(dataset: TSDataset, artifact: ModelArtifact, spec: TaskSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    step = pd.tseries.frequencies.to_offset(spec.freq or "D")
    last_dates = dataset.df.groupby("unique_id")["ds"].max()
    for uid, last_date in last_dates.items():
        for h in range(1, spec.horizon + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": last_date + h * step,
                    "yhat": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_readme_make_plan_tuple_usage() -> None:
    df = _make_panel()
    spec = TaskSpec(h=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    plan, route_decision = make_plan(dataset, spec, use_tsfm=False)

    assert len(plan.candidate_models) > 0
    assert isinstance(route_decision.buckets, list)
    assert isinstance(route_decision.reasons, list)


def test_readme_run_forecast_result_accessors() -> None:
    df = _make_panel()
    spec = TaskSpec(h=7, freq="D")

    result = run_forecast(
        df,
        spec,
        mode="quick",
        fit_func=_fit_stub,
        predict_func=_predict_stub,
    )

    assert not result.forecast.df.empty
    assert isinstance(result.forecast.model_name, str)
    assert isinstance(result.provenance.data_signature, str)
    assert result.provenance.data_signature
