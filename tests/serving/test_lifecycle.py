"""Tests for serving/lifecycle.py."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tsagentkit.contracts import (
    EArtifactLoadFailed,
    EArtifactSchemaIncompatible,
    ForecastResult,
    Provenance,
)
from tsagentkit.router import PlanSpec, compute_plan_signature
from tsagentkit.serving import (
    load_run_artifact,
    package_run,
    replay_forecast_from_artifact,
    save_run_artifact,
    validate_run_artifact_for_serving,
)


def _sample_run_artifact():
    forecast_df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "model": ["Naive", "Naive"],
            "yhat": [1.0, 2.0],
        }
    )
    plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
    provenance = Provenance(
        run_id="lifecycle-run",
        timestamp="2024-01-01T00:00:00Z",
        data_signature="data-sig",
        task_signature="task-sig",
        plan_signature=compute_plan_signature(plan),
        model_signature="model-sig",
    )
    forecast = ForecastResult(
        df=forecast_df,
        provenance=provenance,
        model_name="Naive",
        horizon=2,
    )
    artifact = package_run(
        forecast=forecast,
        plan=plan,
        provenance=provenance,
        metadata={"mode": "test"},
    )
    return artifact, provenance


def test_save_and_load_run_artifact_roundtrip(tmp_path) -> None:
    artifact, _ = _sample_run_artifact()
    output = tmp_path / "run_artifact.json"

    saved = save_run_artifact(artifact, output)
    loaded = load_run_artifact(saved)

    assert loaded.artifact_type == "tsagentkit.run_artifact"
    assert loaded.artifact_schema_version == 1
    assert loaded.forecast.model_name == artifact.forecast.model_name
    assert loaded.forecast.horizon == artifact.forecast.horizon
    assert len(loaded.forecast.df) == len(artifact.forecast.df)


def test_validate_run_artifact_for_serving_signature_gate() -> None:
    artifact, provenance = _sample_run_artifact()

    report = validate_run_artifact_for_serving(
        artifact,
        expected_task_signature=provenance.task_signature,
        expected_plan_signature=provenance.plan_signature,
    )
    assert report["run_id"] == provenance.run_id

    with pytest.raises(EArtifactSchemaIncompatible):
        validate_run_artifact_for_serving(
            artifact,
            expected_task_signature="wrong-task-signature",
        )


def test_replay_forecast_from_artifact_marks_replay_metadata() -> None:
    artifact, provenance = _sample_run_artifact()

    replayed = replay_forecast_from_artifact(artifact)

    assert replayed.model_name == artifact.forecast.model_name
    assert replayed.horizon == artifact.forecast.horizon
    assert replayed.df.equals(artifact.forecast.df)
    assert replayed.provenance.metadata["replayed_from_artifact"] is True
    assert replayed.provenance.metadata["source_run_id"] == provenance.run_id


def test_load_run_artifact_rejects_non_object_payload(tmp_path) -> None:
    output = tmp_path / "bad_artifact.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(["not", "an", "object"], f)

    with pytest.raises(EArtifactLoadFailed):
        load_run_artifact(output)
