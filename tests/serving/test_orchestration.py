"""Tests for serving/orchestration.py."""

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.serving import run_forecast


class TestRunForecast:
    """Tests for run_forecast function."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data."""
        return pd.DataFrame({
            "unique_id": ["A"] * 30 + ["B"] * 30,
            "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
            "y": list(range(30)) * 2,
        })

    @pytest.fixture
    def sample_spec(self) -> TaskSpec:
        """Create sample task spec."""
        return TaskSpec(horizon=7, freq="D")

    def test_quick_mode(self, sample_data: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test quick mode execution."""
        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=sample_data,
            task_spec=sample_spec,
            mode="quick",
            fit_func=fit,
            predict_func=predict,
        )

        assert result is not None
        assert len(result.forecast) > 0
        assert result.model_name is not None
        assert result.metadata["mode"] == "quick"

    def test_standard_mode(self, sample_data: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test standard mode execution."""
        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=sample_data,
            task_spec=sample_spec,
            mode="standard",
            fit_func=fit,
            predict_func=predict,
        )

        assert result is not None
        assert "events" in result.metadata

    def test_creates_provenance(self, sample_data: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test that provenance is created."""
        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=sample_data,
            task_spec=sample_spec,
            mode="quick",
            fit_func=fit,
            predict_func=predict,
        )

        assert "timestamp" in result.provenance
        assert "data_signature" in result.provenance
        assert "task_signature" in result.provenance

    def test_invalid_data_raises(self, sample_spec: TaskSpec) -> None:
        """Test that invalid data raises error."""
        invalid_data = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        })

        from tsagentkit.contracts import EContractMissingColumn

        with pytest.raises(EContractMissingColumn):
            run_forecast(
                data=invalid_data,
                task_spec=sample_spec,
            )

    def test_logs_events(self, sample_data: pd.DataFrame, sample_spec: TaskSpec) -> None:
        """Test that events are logged."""
        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=sample_data,
            task_spec=sample_spec,
            mode="quick",
            fit_func=fit,
            predict_func=predict,
        )

        events = result.metadata.get("events", [])
        event_names = [e["step_name"] for e in events]

        assert "validate" in event_names
        assert "qa" in event_names
        assert "build_dataset" in event_names
        assert "make_plan" in event_names
        assert "fit" in event_names
        assert "predict" in event_names
