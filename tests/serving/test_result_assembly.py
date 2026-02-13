"""Tests for serving result assembly helpers."""

from __future__ import annotations

import pandas as pd

from tsagentkit.contracts import ModelArtifact, PlanSpec, RouteDecision, TaskSpec, ValidationReport
from tsagentkit.qa import QAReport
from tsagentkit.serving.result_assembly import (
    assemble_run_artifact,
    build_dry_run_result,
    resolve_model_identity,
)


def test_build_dry_run_result_characterization() -> None:
    """Dry-run payload should preserve validation/qa/plan serialization shape."""
    plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
    route_decision = RouteDecision(
        stats={"n_series": 1},
        buckets=["default"],
        selected_plan=plan,
        reasons=["fallback"],
    )
    result = build_dry_run_result(
        validation=ValidationReport(valid=True),
        qa_report=QAReport(
            issues=[{"type": "warning", "severity": "warning"}],
            repairs=[{"action": "noop"}],  # characterization: passthrough for non-model objects
            leakage_detected=False,
        ),
        plan=plan,
        route_decision=route_decision,
        task_spec=TaskSpec(h=1, freq="D"),
    )

    assert result.validation["valid"] is True
    assert result.qa_report["issues"][0]["type"] == "warning"
    assert result.qa_report["repairs"][0]["action"] == "noop"
    assert result.plan["plan_name"] == "default"
    assert result.route_decision["buckets"] == ["default"]
    assert result.task_spec_used["h"] == 1


def test_resolve_model_identity_per_series_prefers_selection_map() -> None:
    """Per-series model selection should produce grouped model name."""
    first = ModelArtifact(model=None, model_name="Naive")
    second = ModelArtifact(model=None, model_name="ETS")
    model_name, primary = resolve_model_identity(
        selection_map={"a": "Naive", "b": "ETS"},
        model_artifact=None,
        model_artifacts={"a": first, "b": second},
    )

    assert model_name == "per_series(ETS,Naive)"
    assert primary is first


def test_assemble_run_artifact_single_model_round_trip() -> None:
    """Assembly helper should package a valid RunArtifact for single-model runs."""
    data = pd.DataFrame(
        {
            "unique_id": ["s1", "s1", "s1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [10.0, 12.0, 11.0],
        }
    )
    plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
    task_spec = TaskSpec(h=1, freq="D")
    forecast_df = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2024-01-04"]),
            "model": ["Naive"],
            "yhat": [11.5],
        }
    )
    artifact = assemble_run_artifact(
        data=data,
        task_spec=task_spec,
        plan=plan,
        forecast_df=forecast_df,
        validation=ValidationReport(valid=True),
        backtest_report=None,
        qa_report=None,
        model_artifact=ModelArtifact(model=None, model_name="Naive"),
        model_artifacts=None,
        selection_map=None,
        qa_repairs=[],
        fallbacks_triggered=[],
        feature_matrix=None,
        drift_report=None,
        column_map=None,
        original_panel_contract=task_spec.panel_contract,
        route_decision=None,
        calibration_artifact=None,
        anomaly_report=None,
        degradation_events=[],
        mode="quick",
        total_duration_ms=123.4,
        events=[{"step_name": "predict", "status": "success"}],
    )

    assert artifact.forecast.model_name == "Naive"
    assert artifact.metadata["mode"] == "quick"
    assert artifact.metadata["per_series_selection"] is False
    assert artifact.plan["plan_name"] == "default"
