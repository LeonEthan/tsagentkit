"""Tests for stability monitoring (jitter and coverage)."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.monitoring.report import CalibrationReport, StabilityReport
from tsagentkit.monitoring.stability import (
    StabilityMonitor,
    compute_prediction_interval_coverage,
)


class TestStabilityMonitorInit:
    """Test StabilityMonitor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        monitor = StabilityMonitor()
        assert monitor.jitter_threshold == 0.1
        assert monitor.coverage_tolerance == 0.05

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        monitor = StabilityMonitor(jitter_threshold=0.15, coverage_tolerance=0.02)
        assert monitor.jitter_threshold == 0.15
        assert monitor.coverage_tolerance == 0.02


class TestComputeJitter:
    """Test jitter computation."""

    @pytest.fixture
    def stable_predictions(self):
        """Create predictions with low variance."""
        base = pd.DataFrame({
            "unique_id": ["A"] * 10 + ["B"] * 10,
            "ds": list(pd.date_range("2024-01-01", periods=10)) * 2,
            "yhat": [100.0] * 10 + [200.0] * 10,
        })
        # Slight variations
        preds = []
        for i in range(3):
            p = base.copy()
            p["yhat"] = p["yhat"] + np.random.normal(0, 1, 20)
            preds.append(p)
        return preds

    @pytest.fixture
    def unstable_predictions(self):
        """Create predictions with high variance."""
        base = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "yhat": [100.0] * 10,
        })
        # Large variations
        preds = []
        for i in range(3):
            p = base.copy()
            p["yhat"] = p["yhat"] + np.random.normal(0, 20, 10)
            preds.append(p)
        return preds

    def test_insufficient_predictions(self):
        """Test that single prediction returns empty dict."""
        monitor = StabilityMonitor()
        jitter = monitor.compute_jitter([pd.DataFrame()])
        assert jitter == {}

    def test_cv_method(self, stable_predictions):
        """Test coefficient of variation method."""
        monitor = StabilityMonitor()
        jitter = monitor.compute_jitter(stable_predictions, method="cv")

        assert "A" in jitter
        assert "B" in jitter
        # Stable predictions should have low CV
        assert jitter["A"] < 0.1

    def test_mad_method(self, stable_predictions):
        """Test median absolute deviation method."""
        monitor = StabilityMonitor()
        jitter = monitor.compute_jitter(stable_predictions, method="mad")

        assert "A" in jitter
        # MAD should be small for stable predictions
        assert jitter["A"] < 5

    def test_unstable_predictions_high_jitter(self, unstable_predictions):
        """Test that unstable predictions have high jitter."""
        monitor = StabilityMonitor()
        jitter = monitor.compute_jitter(unstable_predictions, method="cv")

        assert jitter["A"] > 0.1  # High CV for unstable


class TestComputeCoverage:
    """Test quantile coverage computation."""

    @pytest.fixture
    def perfect_quantiles(self):
        """Create perfectly calibrated quantile forecasts."""
        np.random.seed(42)
        actuals = pd.DataFrame({
            "unique_id": ["A"] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "y": np.random.normal(0, 1, 100),
        })

        # For normal distribution, compute proper quantiles
        forecasts = pd.DataFrame({
            "unique_id": ["A"] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "q_0.10": np.percentile(actuals["y"], 10),
            "q_0.50": np.percentile(actuals["y"], 50),
            "q_0.90": np.percentile(actuals["y"], 90),
        })

        return actuals, forecasts

    def test_coverage_computation(self, perfect_quantiles):
        """Test coverage computation."""
        actuals, forecasts = perfect_quantiles
        monitor = StabilityMonitor()

        coverage = monitor.compute_coverage(
            actuals, forecasts, [0.1, 0.5, 0.9]
        )

        assert 0.1 in coverage
        assert 0.5 in coverage
        assert 0.9 in coverage

    def test_coverage_range(self, perfect_quantiles):
        """Test that coverage values are in valid range."""
        actuals, forecasts = perfect_quantiles
        monitor = StabilityMonitor()

        coverage = monitor.compute_coverage(
            actuals, forecasts, [0.1, 0.5, 0.9]
        )

        for q, cov in coverage.items():
            assert 0.0 <= cov <= 1.0


class TestCheckCalibration:
    """Test calibration checking."""

    def test_well_calibrated(self):
        """Test detection of well-calibrated quantiles."""
        np.random.seed(42)
        actuals = pd.DataFrame({
            "unique_id": ["A"] * 1000,
            "ds": pd.date_range("2024-01-01", periods=1000),
            "y": np.random.normal(0, 1, 1000),
        })

        # Compute actual quantiles from data
        q_10 = np.percentile(actuals["y"], 10)
        q_50 = np.percentile(actuals["y"], 50)
        q_90 = np.percentile(actuals["y"], 90)

        forecasts = pd.DataFrame({
            "unique_id": ["A"] * 1000,
            "ds": pd.date_range("2024-01-01", periods=1000),
            "q_0.10": q_10,
            "q_0.50": q_50,
            "q_0.90": q_90,
        })

        monitor = StabilityMonitor(coverage_tolerance=0.05)
        report = monitor.check_calibration(actuals, forecasts, [0.1, 0.5, 0.9])

        assert isinstance(report, CalibrationReport)
        assert report.well_calibrated

    def test_poorly_calibrated(self):
        """Test detection of poorly-calibrated quantiles."""
        np.random.seed(42)
        actuals = pd.DataFrame({
            "unique_id": ["A"] * 1000,
            "ds": pd.date_range("2024-01-01", periods=1000),
            "y": np.random.normal(0, 1, 1000),
        })

        # Intentionally wrong quantiles
        forecasts = pd.DataFrame({
            "unique_id": ["A"] * 1000,
            "ds": pd.date_range("2024-01-01", periods=1000),
            "q_0.10": 5.0,  # Way too high
            "q_0.50": 5.0,
            "q_0.90": 5.0,
        })

        monitor = StabilityMonitor(coverage_tolerance=0.05)
        report = monitor.check_calibration(actuals, forecasts, [0.1, 0.5, 0.9])

        assert not report.well_calibrated


class TestGenerateStabilityReport:
    """Test comprehensive stability report generation."""

    def test_full_report(self):
        """Test generating a full stability report."""
        np.random.seed(42)

        # Create predictions with jitter
        predictions = []
        for i in range(3):
            pred = pd.DataFrame({
                "unique_id": ["A"] * 10,
                "ds": pd.date_range("2024-01-01", periods=10),
                "yhat": np.random.normal(100, 10, 10),
            })
            predictions.append(pred)

        # Create actuals and forecasts for coverage
        actuals = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "y": np.random.normal(100, 10, 10),
        })

        forecasts = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "q_0.50": [100.0] * 10,
        })

        monitor = StabilityMonitor(jitter_threshold=0.5)
        report = monitor.generate_stability_report(
            predictions=predictions,
            actuals=actuals,
            forecasts=forecasts,
            quantiles=[0.5],
        )

        assert isinstance(report, StabilityReport)
        assert "A" in report.jitter_metrics
        assert report.coverage_report is not None

    def test_jitter_only_report(self):
        """Test report with only jitter metrics."""
        predictions = [
            pd.DataFrame({
                "unique_id": ["A"] * 5,
                "ds": pd.date_range("2024-01-01", periods=5),
                "yhat": [100.0, 101.0, 99.0, 100.0, 102.0],
            }),
            pd.DataFrame({
                "unique_id": ["A"] * 5,
                "ds": pd.date_range("2024-01-01", periods=5),
                "yhat": [101.0, 100.0, 100.0, 99.0, 101.0],
            }),
        ]

        monitor = StabilityMonitor()
        report = monitor.generate_stability_report(predictions=predictions)

        assert report.coverage_report is None
        assert "A" in report.jitter_metrics


class TestStabilityReportProperties:
    """Test StabilityReport properties."""

    def test_is_stable_true(self):
        """Test is_stable when no high jitter."""
        report = StabilityReport(
            jitter_metrics={"A": 0.05, "B": 0.08},
            overall_jitter=0.065,
            jitter_threshold=0.1,
            high_jitter_series=[],
        )
        assert report.is_stable

    def test_is_stable_false(self):
        """Test is_stable when high jitter exists."""
        report = StabilityReport(
            jitter_metrics={"A": 0.15, "B": 0.08},
            overall_jitter=0.115,
            jitter_threshold=0.1,
            high_jitter_series=["A"],
        )
        assert not report.is_stable

    def test_summary(self):
        """Test summary method."""
        report = StabilityReport(
            jitter_metrics={"A": 0.05},
            overall_jitter=0.05,
            jitter_threshold=0.1,
            high_jitter_series=[],
        )
        summary = report.summary()
        assert "stable" in summary
        assert "0.050" in summary


class TestComputePredictionIntervalCoverage:
    """Test prediction interval coverage helper."""

    def test_perfect_coverage(self):
        """Test coverage when all actuals within bounds."""
        actuals = pd.Series([5, 6, 7, 8, 9])
        lower = pd.Series([0] * 5)
        upper = pd.Series([10] * 5)

        coverage = compute_prediction_interval_coverage(actuals, lower, upper)
        assert coverage == 1.0

    def test_partial_coverage(self):
        """Test coverage when some actuals outside bounds."""
        actuals = pd.Series([1, 5, 10, 15, 20])
        lower = pd.Series([5] * 5)
        upper = pd.Series([15] * 5)

        coverage = compute_prediction_interval_coverage(actuals, lower, upper)
        assert coverage == 0.6  # 3 out of 5 within bounds

    def test_zero_coverage(self):
        """Test coverage when no actuals within bounds."""
        actuals = pd.Series([1, 2, 3])
        lower = pd.Series([10] * 3)
        upper = pd.Series([20] * 3)

        coverage = compute_prediction_interval_coverage(actuals, lower, upper)
        assert coverage == 0.0
