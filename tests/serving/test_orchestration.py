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
        return TaskSpec(h=7, freq="D")

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
        assert len(result.forecast.df) > 0
        assert result.forecast.model_name is not None
        assert result.metadata["mode"] == "quick"

    def test_infer_freq_when_missing(self, sample_data: pd.DataFrame) -> None:
        """Infer freq when missing and infer_freq=True."""
        from tsagentkit.models import fit, predict

        spec = TaskSpec(h=7, freq=None, infer_freq=True)
        result = run_forecast(
            data=sample_data,
            task_spec=spec,
            mode="quick",
            fit_func=fit,
            predict_func=predict,
        )

        assert result.task_spec is not None
        assert result.task_spec.get("freq") == "D"

    def test_missing_freq_without_infer_raises(self, sample_data: pd.DataFrame) -> None:
        """Raise if freq is missing and infer_freq=False."""
        from tsagentkit.contracts import ETaskSpecInvalid

        spec = TaskSpec(h=7, freq=None, infer_freq=False)
        with pytest.raises(ETaskSpecInvalid):
            run_forecast(
                data=sample_data,
                task_spec=spec,
                mode="quick",
            )

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

    def test_standard_mode_default_backtest_step_uses_horizon(self, sample_data: pd.DataFrame) -> None:
        """When backtest.step is not explicit, step_size should default to horizon."""
        spec = TaskSpec(
            h=3,
            freq="D",
            backtest={"n_windows": 2, "min_train_size": 10},
        )

        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=sample_data,
            task_spec=spec,
            mode="standard",
            fit_func=fit,
            predict_func=predict,
        )

        assert result.backtest_report is not None
        assert result.backtest_report["metadata"]["step_size"] == spec.horizon

    def test_standard_mode_explicit_backtest_step_is_honored(self, sample_data: pd.DataFrame) -> None:
        """When backtest.step is explicit, it should flow into rolling_backtest."""
        spec = TaskSpec(
            h=3,
            freq="D",
            backtest={"n_windows": 2, "min_train_size": 10, "step": 2},
        )

        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=sample_data,
            task_spec=spec,
            mode="standard",
            fit_func=fit,
            predict_func=predict,
        )

        assert result.backtest_report is not None
        assert result.backtest_report["metadata"]["step_size"] == 2

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

        assert result.provenance is not None
        assert result.provenance.timestamp is not None
        assert result.provenance.data_signature is not None
        assert result.provenance.task_signature is not None

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

    def test_leakage_raises_in_strict_mode(self) -> None:
        """Strict mode should raise on observed covariate leakage."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, None],
            "promo": [0, 1, 1],
        })
        spec = TaskSpec(h=1, freq="D", covariate_policy="observed")

        from tsagentkit.contracts import ECovariateLeakage

        with pytest.raises(ECovariateLeakage):
            run_forecast(
                data=df,
                task_spec=spec,
                mode="strict",
            )

    def test_covariate_leakage_fallback_non_strict(self) -> None:
        """Non-strict modes should drop covariates on leakage and continue."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, None],
            "promo": [0, 1, 1],
        })
        spec = TaskSpec(h=1, freq="D", covariate_policy="observed")

        from tsagentkit.models import fit, predict

        for mode in ("quick", "standard"):
            result = run_forecast(
                data=df,
                task_spec=spec,
                mode=mode,
                fit_func=fit,
                predict_func=predict,
            )

            assert result is not None
            assert result.provenance is not None
            assert any(
                f.get("type") == "covariates_dropped"
                for f in (result.provenance.fallbacks_triggered or [])
            )

    def test_predict_failure_triggers_fallback(self, sample_data: pd.DataFrame, monkeypatch) -> None:
        """Predict failures should trigger fallback to the next candidate."""
        from tsagentkit.contracts import ModelArtifact, RouteDecision
        from tsagentkit.router import PlanSpec

        def fake_make_plan(dataset, task_spec, qa_report):
            plan = PlanSpec(plan_name="default", candidate_models=["bad", "good"])
            route_decision = RouteDecision(
                stats={},
                buckets=["test"],
                selected_plan=plan,
                reasons=["test fallback"],
            )
            return plan, route_decision

        monkeypatch.setattr(
            "tsagentkit.serving.orchestration.make_plan",
            fake_make_plan,
        )

        def fit_func(dataset, plan):
            return ModelArtifact(model={}, model_name=plan.candidate_models[0])

        def predict_func(dataset, artifact, spec):
            if artifact.model_name == "bad":
                raise RuntimeError("predict failed")
            rows = []
            step = pd.tseries.frequencies.to_offset(spec.freq)
            for uid, last_date in dataset.df.groupby("unique_id")["ds"].max().items():
                for h in range(1, spec.horizon + 1):
                    rows.append(
                        {
                            "unique_id": uid,
                            "ds": last_date + h * step,
                            "yhat": 1.0,
                        }
                    )
            return pd.DataFrame(rows)

        spec = TaskSpec(h=2, freq="D")
        result = run_forecast(
            data=sample_data,
            task_spec=spec,
            mode="quick",
            fit_func=fit_func,
            predict_func=predict_func,
        )

        assert result.forecast.model_name == "good"
        assert any(
            f.get("from") == "bad" and f.get("to") == "good"
            for f in (result.provenance.fallbacks_triggered or [])
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

    def test_repair_strategy_from_run_forecast(self) -> None:
        """Repairs should use provided repair_strategy when supplied."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, None, 3.0],
        })
        spec = TaskSpec(h=1, freq="D")

        from tsagentkit.models import fit, predict

        result = run_forecast(
            data=df,
            task_spec=spec,
            mode="quick",
            fit_func=fit,
            predict_func=predict,
            repair_strategy={"missing_method": "ffill"},
        )

        assert result.qa_report is not None
        assert any(r.get("repair_type") == "missing_values" for r in result.qa_report.get("repairs", []))
