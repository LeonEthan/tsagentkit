"""Tests for backtest/engine.py."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.backtest import cross_validation_split, rolling_backtest
from tsagentkit.contracts import ESplitRandomForbidden
from tsagentkit.router import Plan
from tsagentkit.series import TSDataset


@pytest.fixture
def sample_dataset() -> TSDataset:
    """Create a sample dataset."""
    df = pd.DataFrame({
        "unique_id": ["A"] * 50 + ["B"] * 50,
        "ds": list(pd.date_range("2024-01-01", periods=50, freq="D")) * 2,
        "y": list(range(50)) * 2,
    })
    spec = TaskSpec(horizon=7, freq="D")
    return TSDataset.from_dataframe(df, spec)


@pytest.fixture
def sample_plan() -> Plan:
    """Create a sample plan."""
    return Plan(
        primary_model="Naive",
        fallback_chain=[],
        config={"season_length": 7},
    )


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
        df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": range(5),
        }).sample(frac=1)  # Shuffle

        with pytest.raises(ESplitRandomForbidden):
            cross_validation_split(df, n_splits=2, horizon=1)


class TestRollingBacktest:
    """Tests for rolling_backtest function."""

    def test_returns_backtest_report(self, sample_dataset: TSDataset, sample_plan: Plan) -> None:
        """Test that rolling_backtest returns a BacktestReport."""
        from tsagentkit.backtest import BacktestReport

        def fit_func(model, data, config):
            return {"type": "naive"}

        def predict_func(model, data, horizon):
            # Return predictions matching test data size
            return pd.DataFrame({
                "unique_id": data["unique_id"],
                "ds": data["ds"],
                "yhat": [1.0] * len(data),
            })

        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=fit_func,
            predict_func=predict_func,
            n_windows=3,
        )

        assert isinstance(report, BacktestReport)

    def test_report_has_windows(self, sample_dataset: TSDataset, sample_plan: Plan) -> None:
        """Test that report contains window results."""
        def fit_func(model, data, config):
            return {"type": "naive"}

        def predict_func(model, data, horizon):
            # Return predictions matching test data size
            return pd.DataFrame({
                "unique_id": data["unique_id"],
                "ds": data["ds"],
                "yhat": [1.0] * len(data),
            })

        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=fit_func,
            predict_func=predict_func,
            n_windows=3,
        )

        assert report.n_windows > 0
        assert len(report.window_results) > 0

    def test_report_has_aggregate_metrics(self, sample_dataset: TSDataset, sample_plan: Plan) -> None:
        """Test that report contains aggregate metrics."""
        def fit_func(model, data, config):
            return {"type": "naive"}

        def predict_func(model, data, horizon):
            result = data[["unique_id", "ds", "y"]].copy()
            result["yhat"] = result["y"]  # Perfect forecast
            return result

        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=fit_func,
            predict_func=predict_func,
            n_windows=2,
        )

        assert "mae" in report.aggregate_metrics


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
        spec = TaskSpec(horizon=3, freq="MS", season_length=12)
        dataset = TSDataset.from_dataframe(df, spec)
        plan = Plan(
            primary_model="Naive",
            fallback_chain=[],
            config={"season_length": 12},
        )

        def fit_func(model, data, config):
            return {"type": model}

        def predict_func(model, data, horizon):
            result = data[["unique_id", "ds", "y"]].copy()
            result["yhat"] = result["y"]
            return result

        report = rolling_backtest(
            dataset=dataset,
            spec=spec,
            plan=plan,
            fit_func=fit_func,
            predict_func=predict_func,
            n_windows=3,
        )

        assert report.n_windows == 3
        assert len(report.window_results) == 3
        assert report.errors == []
        assert report.aggregate_metrics["mae"] <= 1e-9
        assert report.aggregate_metrics["mae"] >= 0

    def test_expanding_strategy(self, sample_dataset: TSDataset, sample_plan: Plan) -> None:
        """Test expanding window strategy."""
        def fit_func(model, data, config):
            return {"type": "naive"}

        def predict_func(model, data, horizon):
            # Return predictions matching test data size
            return pd.DataFrame({
                "unique_id": data["unique_id"],
                "ds": data["ds"],
                "yhat": [1.0] * len(data),
            })

        report = rolling_backtest(
            dataset=sample_dataset,
            spec=sample_dataset.task_spec,
            plan=sample_plan,
            fit_func=fit_func,
            predict_func=predict_func,
            n_windows=2,
            window_strategy="expanding",
        )

        assert report.strategy == "expanding"

    def test_insufficient_data_raises(self, sample_dataset: TSDataset, sample_plan: Plan) -> None:
        """Test that insufficient data raises error."""
        from tsagentkit.contracts import EBacktestInsufficientData

        def fit_func(model, data, config):
            return {"type": "naive"}

        def predict_func(model, data, horizon):
            # Return predictions matching test data size
            return pd.DataFrame({
                "unique_id": data["unique_id"],
                "ds": data["ds"],
                "yhat": [1.0] * len(data),
            })

        with pytest.raises(EBacktestInsufficientData):
            rolling_backtest(
                dataset=sample_dataset,
                spec=sample_dataset.task_spec,
                plan=sample_plan,
                fit_func=fit_func,
                predict_func=predict_func,
                n_windows=100,  # Way too many
            )
