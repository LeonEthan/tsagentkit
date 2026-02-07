"""Tests for backtest/engine.py."""

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.backtest import cross_validation_split, rolling_backtest
from tsagentkit.contracts import ESplitRandomForbidden, ModelArtifact
from tsagentkit.router import PlanSpec
from tsagentkit.series import TSDataset


def _fit_stub(dataset: TSDataset, plan: PlanSpec):
    return ModelArtifact(model={}, model_name=plan.candidate_models[0])


def _predict_stub(dataset: TSDataset, model, spec: TaskSpec) -> pd.DataFrame:
    rows = []
    freq = spec.freq
    step = pd.tseries.frequencies.to_offset(freq)
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


def _predict_fail_stub(dataset: TSDataset, model, spec: TaskSpec) -> pd.DataFrame:
    raise RuntimeError("predict failed")


@pytest.fixture
def sample_dataset() -> TSDataset:
    """Create a sample dataset."""
    df = pd.DataFrame({
        "unique_id": ["A"] * 50 + ["B"] * 50,
        "ds": list(pd.date_range("2024-01-01", periods=50, freq="D")) * 2,
        "y": list(range(50)) * 2,
    })
    spec = TaskSpec(h=7, freq="D")
    return TSDataset.from_dataframe(df, spec)


@pytest.fixture
def sample_plan() -> PlanSpec:
    """Create a sample plan."""
    return PlanSpec(plan_name="default", candidate_models=["Naive"])


class TestValidateTemporalOrdering:
    """Tests for temporal ordering validation."""

    def test_sorted_data_passes(self) -> None:
        """Test that sorted data passes validation."""
        from tsagentkit.backtest.engine import _validate_temporal_ordering

        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        # Should not raise
        _validate_temporal_ordering(df)

    def test_unsorted_data_raises(self) -> None:
        """Test that unsorted data raises error."""
        from tsagentkit.backtest.engine import _validate_temporal_ordering

        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": ["2024-01-02", "2024-01-01"],
            "y": [2.0, 1.0],
        })
        with pytest.raises(ESplitRandomForbidden):
            _validate_temporal_ordering(df)


class TestGenerateCutoffs:
    """Tests for cutoff generation."""

    def test_expanding_window(self) -> None:
        """Test expanding window cutoffs."""
        from tsagentkit.backtest.engine import _generate_cutoffs

        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        cutoffs = _generate_cutoffs(
            dates.tolist(),
            n_windows=3,
            horizon=2,
            step=2,
            min_train_size=5,
            strategy="expanding",
        )

        assert len(cutoffs) == 3
        # First cutoff should be at index 5 (min_train_size)
        assert cutoffs[0][0] == dates[5]

    def test_sliding_window(self) -> None:
        """Test sliding window cutoffs."""
        from tsagentkit.backtest.engine import _generate_cutoffs

        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        cutoffs = _generate_cutoffs(
            dates.tolist(),
            n_windows=3,
            horizon=2,
            step=2,
            min_train_size=5,
            strategy="sliding",
        )

        assert len(cutoffs) == 3


class TestCrossValidationSplit:
    """Tests for cross_validation_split function."""

    def test_creates_temporal_splits(self) -> None:
        """Test that splits are temporal."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": range(10),
        })

        splits = cross_validation_split(df, n_splits=3, horizon=2)

        assert len(splits) > 0
        for train_df, test_df in splits:
            # Train dates should all be before test dates
            assert train_df["ds"].max() < test_df["ds"].min()

    def test_raises_on_shuffled_data(self) -> None:
        """Test that shuffled data raises error."""
        base_df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": range(5),
        })
        df = base_df.iloc[[0, 2, 1, 3, 4]].reset_index(drop=True)

        with pytest.raises(ESplitRandomForbidden):
            cross_validation_split(df, n_splits=2, horizon=1)


class TestRollingBacktest:
    """Tests for rolling_backtest function."""

    def test_returns_backtest_report(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Test that rolling_backtest returns a BacktestReport."""
        from tsagentkit.backtest import BacktestReport

        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=3,
        )

        assert isinstance(report, BacktestReport)

    def test_report_has_windows(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Test that report contains window results."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=3,
        )

        assert report.n_windows > 0
        assert len(report.window_results) > 0

    def test_default_step_size_uses_horizon(
        self,
        sample_dataset: TSDataset,
        sample_plan: PlanSpec,
    ) -> None:
        """Default step size should match forecast horizon."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=2,
            min_train_size=20,
        )

        assert report.metadata["step_size"] == sample_dataset.task_spec.horizon

    def test_custom_step_size_is_honored(
        self,
        sample_dataset: TSDataset,
        sample_plan: PlanSpec,
    ) -> None:
        """Explicit step_size should be preserved in metadata and windows."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=3,
            min_train_size=20,
            step_size=2,
        )

        assert report.metadata["step_size"] == 2
        assert report.n_windows == 3

        test_starts = [pd.Timestamp(w.test_start) for w in report.window_results]
        assert (test_starts[1] - test_starts[0]).days == 2

    def test_report_has_aggregate_metrics(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Test that report contains aggregate metrics."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=2,
        )

        assert "mae" in report.aggregate_metrics

    def test_report_has_decision_summary(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Report metadata should include a decision summary."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=1,
        )

        decision = report.metadata.get("decision_summary", {})
        assert decision.get("plan_name") == sample_plan.plan_name
        assert decision.get("primary_model") == sample_plan.candidate_models[0]

    def test_predict_errors_include_model_and_stage(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Predict failures should be recorded with model/stage details."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_fail_stub,
            n_windows=1,
        )

        assert any(
            e.get("stage") == "predict" and e.get("model") == sample_plan.candidate_models[0]
            for e in report.errors
        )


class TestRollingBacktestRealWorldData:
    """Tests using a real-world dataset (AirPassengers)."""

    @staticmethod
    def _air_passengers_dataframe() -> pd.DataFrame:
        """Return a real-world monthly passenger dataset (1949-1951)."""
        values = [
            112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
            115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
            145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        ]
        dates = pd.date_range("1949-01-01", periods=len(values), freq="MS")
        return pd.DataFrame({
            "unique_id": ["AirPassengers"] * len(values),
            "ds": dates,
            "y": values,
        })

    def test_backtest_runs_on_air_passengers(self) -> None:
        """Ensure rolling backtest works on real-world data."""
        df = self._air_passengers_dataframe()
        spec = TaskSpec(h=3, freq="MS")
        dataset = TSDataset.from_dataframe(df, spec)
        plan = PlanSpec(plan_name="default", candidate_models=["Naive"])

        report = rolling_backtest(
            dataset=dataset,
            spec=spec,
            plan=plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=3,
        )

        assert report.n_windows == 3
        assert len(report.window_results) == 3
        assert report.errors == []
        assert report.aggregate_metrics["mae"] >= 0

    def test_expanding_strategy(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Test expanding window strategy."""
        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=_fit_stub,
            predict_func=_predict_stub,
            n_windows=2,
            window_strategy="expanding",
        )

        assert report.strategy == "expanding"

    def test_insufficient_data_raises(self, sample_dataset: TSDataset, sample_plan: PlanSpec) -> None:
        """Test that insufficient data raises error."""
        from tsagentkit.contracts import EBacktestInsufficientData

        with pytest.raises(EBacktestInsufficientData):
            rolling_backtest(
                dataset=sample_dataset,
                spec=sample_dataset.task_spec,
                plan=sample_plan,
                fit_func=_fit_stub,
                predict_func=_predict_stub,
                n_windows=100,  # Way too many
            )
