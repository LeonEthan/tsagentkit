"""Tests for contracts/artifact_payloads.py."""

from __future__ import annotations

import pandas as pd

from tsagentkit.anomaly import AnomalyReport
from tsagentkit.calibration import CalibratorArtifact
from tsagentkit.contracts import (
    AnomalyReportPayload,
    CalibrationArtifactPayload,
    RunArtifactPayload,
    anomaly_payload_from_any,
    calibration_payload_from_any,
    run_artifact_payload_from_dict,
)


def test_calibration_payload_from_dataclass() -> None:
    artifact = CalibratorArtifact(
        method="conformal",
        level=95,
        by="global",
        deltas={"global": 1.25},
        metadata={"n_residuals": 10},
    )

    payload = calibration_payload_from_any(artifact)
    assert isinstance(payload, CalibrationArtifactPayload)
    assert payload.method == "conformal"
    assert payload.level == 95
    assert payload.deltas["global"] == 1.25


def test_anomaly_payload_from_dataclass_with_frame() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
            "yhat": [1.1],
            "lo": [0.9],
            "hi": [1.3],
            "anomaly": [False],
            "anomaly_score": [0.0],
            "threshold": [95],
            "method": ["interval_breach"],
            "score": ["normalized_margin"],
        }
    )
    report = AnomalyReport(
        frame=frame,
        method="interval_breach",
        level=95,
        score="normalized_margin",
        summary={"total": 1, "anomalies": 0, "anomaly_rate": 0.0},
    )

    payload = anomaly_payload_from_any(report)
    assert isinstance(payload, AnomalyReportPayload)
    assert payload.method == "interval_breach"
    assert payload.level == 95
    assert len(payload.frame) == 1
    assert payload.frame[0]["unique_id"] == "A"


def test_run_artifact_payload_validation_with_nested_payloads() -> None:
    payload = run_artifact_payload_from_dict(
        {
            "forecast": {"model_name": "Naive", "horizon": 2, "df": []},
            "plan": {"candidate_models": ["Naive"]},
            "metadata": {"mode": "quick"},
            "calibration_artifact": {
                "method": "none",
                "level": 95,
                "by": "global",
                "deltas": {"global": 0.0},
                "metadata": {},
            },
            "anomaly_report": {
                "method": "interval_breach",
                "level": 95,
                "score": "normalized_margin",
                "summary": {"total": 0, "anomalies": 0, "anomaly_rate": 0.0},
                "frame": [],
            },
        }
    )

    assert isinstance(payload, RunArtifactPayload)
    assert payload.calibration_artifact is not None
    assert payload.anomaly_report is not None
