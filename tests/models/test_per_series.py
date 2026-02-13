"""Tests for per-series fit/predict helpers."""

from __future__ import annotations

import pandas as pd

from tsagentkit.contracts import ModelArtifact, TaskSpec
from tsagentkit.models.per_series import fit_predict_per_series
from tsagentkit.router import PlanSpec


class _DatasetStub:
    def __init__(self, series_ids: list[str]) -> None:
        self.series_ids = list(series_ids)

    def filter_series(self, series_ids: list[str]) -> _DatasetStub:
        return _DatasetStub(series_ids)


def test_fit_predict_per_series_with_fit_func_without_on_fallback_kwarg() -> None:
    dataset = _DatasetStub(["series-a", "series-b"])
    plan = PlanSpec(
        plan_name="per-series-regression",
        candidate_models=["model-a", "model-b"],
        quantiles=[0.5],
    )
    spec = TaskSpec(h=1, freq="D")
    selection_map = {
        "series-a": "model-a",
        "series-b": "model-b",
    }

    def fit_without_on_fallback(subset: _DatasetStub, model_plan: PlanSpec) -> ModelArtifact:
        _ = subset
        return ModelArtifact(
            model=object(),
            model_name=model_plan.candidate_models[0],
            config={"quantiles": [0.5]},
        )

    def predict_stub(dataset: _DatasetStub, artifact: ModelArtifact, spec: TaskSpec) -> pd.DataFrame:
        _ = spec
        return pd.DataFrame(
            {
                "unique_id": dataset.series_ids,
                "ds": [pd.Timestamp("2024-01-01")] * len(dataset.series_ids),
                "yhat": [1.0] * len(dataset.series_ids),
                "model": [artifact.model_name] * len(dataset.series_ids),
            }
        )

    artifacts, forecast_df = fit_predict_per_series(
        dataset=dataset,
        plan=plan,
        selection_map=selection_map,
        spec=spec,
        fit_func=fit_without_on_fallback,
        predict_func=predict_stub,
        on_fallback=lambda _from, _to, _err: None,
    )

    assert set(artifacts.keys()) == {"model-a", "model-b"}
    assert set(forecast_df["unique_id"]) == {"series-a", "series-b"}
    assert set(forecast_df["model"]) == {"model-a", "model-b"}
