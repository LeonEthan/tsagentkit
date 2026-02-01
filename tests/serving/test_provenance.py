"""Tests for serving/provenance.py."""

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.router import Plan
from tsagentkit.serving.provenance import (
    compute_config_signature,
    compute_data_signature,
    create_provenance,
    log_event,
)


class TestComputeDataSignature:
    """Tests for compute_data_signature function."""

    def test_consistent_hash(self) -> None:
        """Test that same data produces same hash."""
        df1 = pd.DataFrame({
            "unique_id": ["A", "A", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "y": [1.0, 2.0, 3.0],
        })
        df2 = pd.DataFrame({
            "unique_id": ["A", "A", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "y": [1.0, 2.0, 3.0],
        })

        sig1 = compute_data_signature(df1)
        sig2 = compute_data_signature(df2)

        assert sig1 == sig2
        assert len(sig1) == 16

    def test_different_data_different_hash(self) -> None:
        """Test that different data produces different hash."""
        df1 = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
        })
        df2 = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [2.0],
        })

        sig1 = compute_data_signature(df1)
        sig2 = compute_data_signature(df2)

        assert sig1 != sig2


class TestComputeConfigSignature:
    """Tests for compute_config_signature function."""

    def test_consistent_hash(self) -> None:
        """Test that same config produces same hash."""
        config1 = {"season_length": 7, "horizon": 14}
        config2 = {"season_length": 7, "horizon": 14}

        sig1 = compute_config_signature(config1)
        sig2 = compute_config_signature(config2)

        assert sig1 == sig2

    def test_different_config_different_hash(self) -> None:
        """Test that different config produces different hash."""
        config1 = {"season_length": 7}
        config2 = {"season_length": 14}

        sig1 = compute_config_signature(config1)
        sig2 = compute_config_signature(config2)

        assert sig1 != sig2

    def test_empty_config(self) -> None:
        """Test hashing empty config."""
        sig = compute_config_signature({})
        assert len(sig) == 16


class TestCreateProvenance:
    """Tests for create_provenance function."""

    def test_creates_provenance(self) -> None:
        """Test that provenance object is created."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
        })
        spec = TaskSpec(horizon=7, freq="D")
        plan = Plan(primary_model="Naive")

        provenance = create_provenance(df, spec, plan)

        assert provenance.timestamp is not None
        assert provenance.data_signature is not None
        assert provenance.task_signature is not None
        assert provenance.plan_signature is not None
        assert provenance.model_signature is not None
        assert provenance.qa_repairs == []
        assert provenance.fallbacks_triggered == []

    def test_includes_repairs_and_fallbacks(self) -> None:
        """Test that repairs and fallbacks are included."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
        })
        spec = TaskSpec(horizon=7, freq="D")
        plan = Plan(primary_model="Naive")

        repairs = [{"type": "interpolate"}]
        fallbacks = [{"from": "A", "to": "B"}]

        provenance = create_provenance(
            df,
            spec,
            plan,
            qa_repairs=repairs,
            fallbacks_triggered=fallbacks,
        )

        assert provenance.qa_repairs == repairs
        assert provenance.fallbacks_triggered == fallbacks


class TestLogEvent:
    """Tests for log_event function."""

    def test_creates_event_dict(self) -> None:
        """Test that event dict is created."""
        event = log_event(
            step_name="validate",
            status="success",
            duration_ms=100.0,
        )

        assert event["step_name"] == "validate"
        assert event["status"] == "success"
        assert event["duration_ms"] == 100.0
        assert event["error_code"] is None
        assert event["artifacts_generated"] == []

    def test_includes_error_code(self) -> None:
        """Test that error code is included."""
        event = log_event(
            step_name="fit",
            status="failed",
            duration_ms=50.0,
            error_code="E_MODEL_FIT_FAILED",
        )

        assert event["status"] == "failed"
        assert event["error_code"] == "E_MODEL_FIT_FAILED"

    def test_includes_artifacts(self) -> None:
        """Test that artifacts are included."""
        event = log_event(
            step_name="fit",
            status="success",
            duration_ms=200.0,
            artifacts_generated=["model", "report"],
        )

        assert event["artifacts_generated"] == ["model", "report"]
