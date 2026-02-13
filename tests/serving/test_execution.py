"""Tests for serving model execution helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit.contracts import ForecastResult, Provenance, TaskSpec
from tsagentkit.serving.execution import (
    fit_predict_with_fallback,
    fit_single_model,
    predict_single_model,
)


def test_fit_single_model_custom_func_filters_unsupported_kwargs() -> None:
    """Custom fit functions without covariate kwargs should still run."""
    captured: dict[str, object] = {}

    def custom_fit(dataset, plan):  # noqa: ANN001
        captured["dataset"] = dataset
        captured["plan"] = plan
        return {"ok": True}

    artifact = fit_single_model(
        dataset="dataset",
        plan="plan",
        fit_func=custom_fit,
        on_fallback="unused",
        covariates="covariates",
    )

    assert artifact == {"ok": True}
    assert captured == {"dataset": "dataset", "plan": "plan"}


def test_fit_single_model_default_path_passes_on_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default fit path should pass on_fallback and covariates through."""
    import tsagentkit.models as models_module

    captured: dict[str, object] = {}

    def fake_fit(dataset, plan, on_fallback=None, covariates=None):  # noqa: ANN001
        captured["args"] = (dataset, plan, on_fallback, covariates)
        return {"artifact": True}

    monkeypatch.setattr(models_module, "fit", fake_fit)

    artifact = fit_single_model(
        dataset="dataset",
        plan="plan",
        fit_func=None,
        on_fallback="callback",
        covariates="covariates",
    )

    assert artifact == {"artifact": True}
    assert captured["args"] == ("dataset", "plan", "callback", "covariates")


def test_predict_single_model_unwraps_forecast_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predict helper should unwrap ForecastResult to dataframe."""
    import tsagentkit.models as models_module

    forecast_df = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "model": ["Naive"],
            "yhat": [1.0],
        }
    )
    provenance = Provenance(
        run_id="r1",
        timestamp="2024-01-01T00:00:00+00:00",
        data_signature="d",
        task_signature="t",
        plan_signature="p",
        model_signature="m",
    )

    def fake_predict(dataset, artifact, spec, covariates=None):  # noqa: ANN001
        _ = dataset, artifact, spec, covariates
        return ForecastResult(
            df=forecast_df,
            provenance=provenance,
            model_name="Naive",
            horizon=1,
        )

    monkeypatch.setattr(models_module, "predict", fake_predict)

    result = predict_single_model(
        dataset="dataset",
        model_artifact="artifact",
        task_spec=TaskSpec(h=1, freq="D"),
        predict_func=None,
        covariates="covariates",
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["unique_id", "ds", "model", "yhat"]


def test_fit_predict_with_fallback_delegates_to_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback helper should forward args to router fallback function."""
    import tsagentkit.router.fallback as fallback_module

    captured: dict[str, object] = {}
    expected_forecast = pd.DataFrame({"unique_id": ["s1"], "ds": [pd.Timestamp("2024-01-01")], "yhat": [1.0]})

    def fake_fallback(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return "artifact", expected_forecast

    monkeypatch.setattr(fallback_module, "fit_predict_with_fallback", fake_fallback)

    artifact, forecast = fit_predict_with_fallback(
        dataset="dataset",
        plan="plan",
        task_spec="spec",
        fit_func="fit_func",
        predict_func="predict_func",
        covariates="covariates",
        start_after="model_a",
        initial_error=RuntimeError("boom"),
        on_fallback="callback",
        reconciliation_method="bottom_up",
    )

    assert artifact == "artifact"
    assert forecast.equals(expected_forecast)
    assert captured["start_after"] == "model_a"
    assert captured["reconciliation_method"] == "bottom_up"
