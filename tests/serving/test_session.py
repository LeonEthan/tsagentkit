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


def test_session_fit_and_predict_delegate_to_step_functions(monkeypatch) -> None:
    import tsagentkit.serving.orchestration as orch

    captured: dict[str, object] = {}

    def fake_step_fit(dataset, plan, fit_func, on_fallback=None, covariates=None):
        captured["fit_args"] = (dataset, plan, fit_func, on_fallback, covariates)
        return {"artifact": True}

    def fake_step_predict(
        artifact,
        dataset,
        task_spec,
        predict_func,
        plan=None,
        covariates=None,
        reconciliation_method="bottom_up",
    ):
        captured["predict_args"] = (
            artifact,
            dataset,
            task_spec,
            predict_func,
            plan,
            covariates,
            reconciliation_method,
        )
        return pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-01-01")], "yhat": [1.0]})

    monkeypatch.setattr(orch, "_step_fit", fake_step_fit)
    monkeypatch.setattr(orch, "_step_predict", fake_step_predict)

    session = TSAgentSession()
    artifact = session.fit(
        dataset="dataset",
        plan="plan",
        fit_func="fit_func",
        on_fallback="cb",
        covariates="covariates",
    )
    forecast_df = session.predict(
        artifact=artifact,
        dataset="dataset",
        task_spec=_sample_spec(),
        predict_func="predict_func",
        plan="plan",
        covariates="covariates",
        reconciliation_method="min_trace",
    )

    assert artifact == {"artifact": True}
    assert captured["fit_args"] == ("dataset", "plan", "fit_func", "cb", "covariates")
    assert captured["predict_args"][0] == {"artifact": True}
    assert captured["predict_args"][1] == "dataset"
    assert captured["predict_args"][3] == "predict_func"
    assert captured["predict_args"][6] == "min_trace"
    assert not forecast_df.empty


def test_session_run_quick_mode_end_to_end() -> None:
    session = TSAgentSession(mode="quick")
    result = session.run(
        data=_sample_data(),
        task_spec=_sample_spec(),
    )

    assert result is not None
    assert not result.forecast.df.empty
    assert result.metadata["mode"] == "quick"

