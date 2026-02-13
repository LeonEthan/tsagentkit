"""Tests for TSAgentSession and run_forecast compatibility wrapper."""

from __future__ import annotations

import pandas as pd

from tsagentkit import TaskSpec
from tsagentkit.serving.orchestration import TSAgentSession, run_forecast


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["A"] * 20 + ["B"] * 20,
            "ds": list(pd.date_range("2024-01-01", periods=20, freq="D")) * 2,
            "y": list(range(20)) + list(range(20, 0, -1)),
        }
    )


def _sample_spec() -> TaskSpec:
    return TaskSpec(h=3, freq="D", tsfm_policy={"mode": "preferred"})


def test_run_forecast_wrapper_creates_temporary_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, **kwargs):
        captured["self"] = self
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(TSAgentSession, "run", fake_run)
    result = run_forecast(
        data=_sample_data(),
        task_spec=_sample_spec(),
        mode="quick",
    )

    assert result == "sentinel"
    assert isinstance(captured["self"], TSAgentSession)
    assert captured["self"].mode == "quick"
    assert captured["kwargs"]["mode"] == "quick"


def test_run_forecast_wrapper_uses_provided_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(self, **kwargs):
        captured["self"] = self
        captured["kwargs"] = kwargs
        return "provided"

    monkeypatch.setattr(TSAgentSession, "run", fake_run)

    session = TSAgentSession(mode="strict")
    result = run_forecast(
        data=_sample_data(),
        task_spec=_sample_spec(),
        mode="quick",
        session=session,
    )

    assert result == "provided"
    assert captured["self"] is session
    assert captured["kwargs"]["mode"] == "quick"


def test_session_fit_and_predict_delegate_to_models(monkeypatch) -> None:
    """Test that TSAgentSession.fit() and predict() call the models module correctly."""
    import tsagentkit.models as models_module

    captured: dict[str, object] = {}

    def fake_fit(dataset, plan, on_fallback=None, covariates=None):
        captured["fit_args"] = (dataset, plan, on_fallback, covariates)
        return {"artifact": True, "model_name": "test_model"}

    def fake_predict(dataset, artifact, spec, covariates=None):
        captured["predict_args"] = (dataset, artifact, spec, covariates)
        return pd.DataFrame({
            "unique_id": ["A"],
            "ds": [pd.Timestamp("2024-01-01")],
            "yhat": [1.0],
        })

    monkeypatch.setattr(models_module, "fit", fake_fit)
    monkeypatch.setattr(models_module, "predict", fake_predict)

    session = TSAgentSession()
    artifact = session.fit(
        dataset="dataset",
        plan="plan",
        fit_func=None,  # Use default (which is now monkeypatched)
        on_fallback="cb",
        covariates="covariates",
    )

    assert artifact == {"artifact": True, "model_name": "test_model"}
    assert captured["fit_args"] == ("dataset", "plan", "cb", "covariates")


def test_session_run_quick_mode_end_to_end() -> None:
    session = TSAgentSession(mode="quick")
    result = session.run(
        data=_sample_data(),
        task_spec=_sample_spec(),
    )

    assert result is not None
    assert not result.forecast.df.empty
    assert result.metadata["mode"] == "quick"

