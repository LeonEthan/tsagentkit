"""Tests for serving/provenance.py."""

import json

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.router import Plan
from tsagentkit.serving.provenance import (
    StructuredLogger,
    compute_config_signature,
    compute_data_signature,
    create_provenance,
    format_event_json,
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

    def test_includes_timestamp(self) -> None:
        """Test that timestamp is included."""
        event = log_event(
            step_name="validate",
            status="success",
            duration_ms=100.0,
        )

        assert "timestamp" in event
        assert isinstance(event["timestamp"], str)

    def test_includes_context(self) -> None:
        """Test that context is included."""
        event = log_event(
            step_name="qa",
            status="success",
            duration_ms=150.0,
            context={"issues_found": 3, "repairs_applied": 2},
        )

        assert event["context"]["issues_found"] == 3
        assert event["context"]["repairs_applied"] == 2


class TestFormatEventJson:
    """Tests for format_event_json function."""

    def test_formats_event_as_json(self) -> None:
        """Test that event is formatted as JSON."""
        event = log_event(
            step_name="validate",
            status="success",
            duration_ms=100.0,
        )

        json_str = format_event_json(event)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["step_name"] == "validate"

    def test_json_is_deterministic(self) -> None:
        """Test that JSON output is deterministic."""
        event = log_event(
            step_name="validate",
            status="success",
            duration_ms=100.0,
        )

        json_str1 = format_event_json(event)
        json_str2 = format_event_json(event)

        assert json_str1 == json_str2


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_basic_logging(self) -> None:
        """Test basic logging functionality."""
        logger = StructuredLogger()

        event = logger.log("step1", "success", 100.0)

        assert event["step_name"] == "step1"
        assert len(logger.events) == 1

    def test_start_end_pattern(self) -> None:
        """Test start/end pattern."""
        logger = StructuredLogger()

        logger.start_step("fit")
        import time

        time.sleep(0.01)  # Small delay
        event = logger.end_step("fit", status="success")

        assert event["step_name"] == "fit"
        assert event["duration_ms"] > 0

    def test_multiple_events(self) -> None:
        """Test multiple events."""
        logger = StructuredLogger()

        logger.log("validate", "success", 50.0)
        logger.log("qa", "success", 100.0)
        logger.log("fit", "success", 500.0)

        assert len(logger.events) == 3

        summary = logger.get_summary()
        assert summary["total_events"] == 3
        assert summary["success_count"] == 3
        assert summary["failed_count"] == 0
        assert summary["total_duration_ms"] == 650.0

    def test_to_json_export(self) -> None:
        """Test JSON export."""
        logger = StructuredLogger()

        logger.log("validate", "success", 50.0)
        logger.log("fit", "failed", 100.0, error_code="E_ERROR")

        json_str = logger.to_json()
        parsed = json.loads(json_str)

        assert len(parsed) == 2
        assert parsed[0]["step_name"] == "validate"
        assert parsed[1]["error_code"] == "E_ERROR"

    def test_to_dict_export(self) -> None:
        """Test dict export returns copy."""
        logger = StructuredLogger()

        logger.log("validate", "success", 50.0)

        events = logger.to_dict()
        assert len(events) == 1

        # Should be a copy
        events.append({})
        assert len(logger.events) == 1

    def test_failure_counting(self) -> None:
        """Test failure counting in summary."""
        logger = StructuredLogger()

        logger.log("step1", "success", 50.0)
        logger.log("step2", "failed", 100.0)
        logger.log("step3", "failed", 200.0, error_code="E_ERROR")

        summary = logger.get_summary()
        assert summary["success_count"] == 1
        assert summary["failed_count"] == 2

    def test_end_without_start(self) -> None:
        """Test ending step that was never started."""
        logger = StructuredLogger()

        event = logger.end_step("unknown", status="success")

        # Should create event with 0 duration
        assert event["duration_ms"] == 0.0
