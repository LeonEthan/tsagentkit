"""Tests for serving/packaging.py."""

import pandas as pd
import pytest

from tsagentkit.contracts import ModelArtifact
from tsagentkit.router import Plan
from tsagentkit.serving import RunArtifact, package_run


class TestRunArtifact:
    """Tests for RunArtifact dataclass."""

    @pytest.fixture
    def sample_artifact(self) -> RunArtifact:
        """Create a sample RunArtifact."""
        forecast = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "yhat": [1.0, 2.0],
        })
        plan = Plan(primary_model="Naive")

        return RunArtifact(
            forecast=forecast,
            plan=plan,
            model_name="Naive",
        )

    def test_creation(self, sample_artifact: RunArtifact) -> None:
        """Test creating a RunArtifact."""
        assert len(sample_artifact.forecast) == 2
        assert sample_artifact.model_name == "Naive"

    def test_to_dict(self, sample_artifact: RunArtifact) -> None:
        """Test conversion to dictionary."""
        d = sample_artifact.to_dict()

        assert "forecast" in d
        assert d["model_name"] == "Naive"
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
            "yhat": [1.0],
        })
        plan = Plan(primary_model="Naive")

        artifact = package_run(
            forecast=forecast,
            plan=plan,
            model_name="Naive",
        )

        assert isinstance(artifact, RunArtifact)
        assert artifact.model_name == "Naive"

    def test_includes_optional_fields(self) -> None:
        """Test that optional fields are included."""
        forecast = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "yhat": [1.0],
        })
        plan = Plan(primary_model="Naive")
        model_artifact = ModelArtifact(model={}, model_name="Naive")
        provenance = {"timestamp": "2024-01-01"}
        metadata = {"mode": "standard"}

        artifact = package_run(
            forecast=forecast,
            plan=plan,
            model_name="Naive",
            model_artifact=model_artifact,
            provenance=provenance,
            metadata=metadata,
        )

        assert artifact.model_artifact is not None
        assert artifact.provenance == provenance
        assert artifact.metadata == metadata
