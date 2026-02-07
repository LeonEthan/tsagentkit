"""Tests for serving/packaging.py."""

import pandas as pd
import pytest

from tsagentkit.anomaly import AnomalyReport
from tsagentkit.calibration import CalibratorArtifact
from tsagentkit.contracts import ForecastResult, ModelArtifact, Provenance
from tsagentkit.router import PlanSpec, compute_plan_signature
from tsagentkit.serving import RunArtifact, package_run


class TestRunArtifact:
    """Tests for RunArtifact dataclass."""

    @pytest.fixture
    def sample_artifact(self) -> RunArtifact:
        """Create a sample RunArtifact."""
        forecast = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "model": ["Naive", "Naive"],
            "yhat": [1.0, 2.0],
        })
        plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
        provenance = Provenance(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="sig1",
            task_signature="sig2",
            plan_signature=compute_plan_signature(plan),
            model_signature="sig3",
        )
        forecast_result = ForecastResult(
            df=forecast,
            provenance=provenance,
            model_name="Naive",
            horizon=1,
        )

        return RunArtifact(
            forecast=forecast_result,
            plan=plan.model_dump(),
            model_artifact=None,
            provenance=provenance,
        )

    def test_creation(self, sample_artifact: RunArtifact) -> None:
        """Test creating a RunArtifact."""
        assert len(sample_artifact.forecast.df) == 2
        assert sample_artifact.forecast.model_name == "Naive"

    def test_to_dict(self, sample_artifact: RunArtifact) -> None:
        """Test conversion to dictionary."""
        d = sample_artifact.to_dict()

        assert "forecast" in d
        assert d["forecast"]["model_name"] == "Naive"
        assert "plan" in d
        assert "provenance" in d

    def test_summary(self, sample_artifact: RunArtifact) -> None:
        """Test summary generation."""
        summary = sample_artifact.summary()

        assert "Naive" in summary
        assert "Forecast rows" in summary


class TestPackageRun:
    """Tests for package_run function."""

    def test_creates_artifact(self) -> None:
        """Test that artifact is created."""
        forecast = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "model": ["Naive"],
            "yhat": [1.0],
        })
        plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
        provenance = Provenance(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="sig1",
            task_signature="sig2",
            plan_signature=compute_plan_signature(plan),
            model_signature="sig3",
        )
        forecast_result = ForecastResult(
            df=forecast,
            provenance=provenance,
            model_name="Naive",
            horizon=1,
        )

        artifact = package_run(
            forecast=forecast_result,
            plan=plan,
        )

        assert isinstance(artifact, RunArtifact)
        assert artifact.forecast.model_name == "Naive"

    def test_includes_optional_fields(self) -> None:
        """Test that optional fields are included."""
        forecast = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "model": ["Naive"],
            "yhat": [1.0],
        })
        plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
        model_artifact = ModelArtifact(model={}, model_name="Naive")
        provenance = Provenance(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="sig1",
            task_signature="sig2",
            plan_signature=compute_plan_signature(plan),
            model_signature="sig3",
        )
        forecast_result = ForecastResult(
            df=forecast,
            provenance=provenance,
            model_name="Naive",
            horizon=1,
        )
        metadata = {"mode": "standard"}

        artifact = package_run(
            forecast=forecast_result,
            plan=plan,
            model_artifact=model_artifact,
            provenance=provenance,
            metadata=metadata,
        )

        assert artifact.model_artifact is not None
        assert artifact.provenance == provenance
        assert artifact.metadata == metadata

    def test_serializes_calibration_and_anomaly_payloads(self) -> None:
        """Non-dict calibration/anomaly inputs should be serialized to dict payloads."""
        forecast = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "model": ["Naive"],
            "yhat": [1.0],
        })
        plan = PlanSpec(plan_name="default", candidate_models=["Naive"])
        provenance = Provenance(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="sig1",
            task_signature="sig2",
            plan_signature=compute_plan_signature(plan),
            model_signature="sig3",
        )
        forecast_result = ForecastResult(
            df=forecast,
            provenance=provenance,
            model_name="Naive",
            horizon=1,
        )

        calibration_artifact = CalibratorArtifact(
            method="conformal",
            level=95,
            by="global",
            deltas={"global": 1.0},
            metadata={"n_residuals": 5},
        )
        anomaly_frame = pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": pd.to_datetime(["2024-01-01"]),
                "y": [1.0],
                "yhat": [1.0],
                "lo": [0.8],
                "hi": [1.2],
                "anomaly": [False],
                "anomaly_score": [0.0],
                "threshold": [95],
                "method": ["interval_breach"],
                "score": ["normalized_margin"],
            }
        )
        anomaly_report = AnomalyReport(
            frame=anomaly_frame,
            method="interval_breach",
            level=95,
            score="normalized_margin",
            summary={"total": 1, "anomalies": 0, "anomaly_rate": 0.0},
        )

        artifact = package_run(
            forecast=forecast_result,
            plan=plan,
            calibration_artifact=calibration_artifact,
            anomaly_report=anomaly_report,
        )

        assert artifact.calibration_artifact is not None
        assert artifact.calibration_artifact["method"] == "conformal"
        assert artifact.anomaly_report is not None
        assert artifact.anomaly_report["method"] == "interval_breach"
        assert isinstance(artifact.anomaly_report["frame"], list)

        serialized = artifact.to_dict()
        assert serialized["calibration_artifact"]["level"] == 95
        assert serialized["anomaly_report"]["score"] == "normalized_margin"
