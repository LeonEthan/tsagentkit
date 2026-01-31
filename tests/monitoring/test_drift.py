"""Tests for drift detection (PSI and KS)."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.monitoring.drift import DriftDetector, compute_psi_summary
from tsagentkit.monitoring.report import DriftReport


class TestDriftDetectorInit:
    """Test DriftDetector initialization."""

    def test_default_psi_detector(self):
        """Test default PSI detector creation."""
        detector = DriftDetector()
        assert detector.method == "psi"
        assert detector.threshold == 0.2
        assert detector.n_bins == 10

    def test_default_ks_detector(self):
        """Test default KS detector creation."""
        detector = DriftDetector(method="ks")
        assert detector.method == "ks"
        assert detector.threshold == 0.05
        assert detector.n_bins == 10

    def test_custom_threshold(self):
        """Test detector with custom threshold."""
        detector = DriftDetector(method="psi", threshold=0.15)
        assert detector.threshold == 0.15

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Method must be 'psi' or 'ks'"):
            DriftDetector(method="invalid")


class TestPSIDetection:
    """Test PSI drift detection."""

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset."""
        np.random.seed(42)
        return pd.DataFrame({
            "unique_id": ["A"] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "sales": np.random.normal(100, 15, 100),
            "price": np.random.normal(50, 5, 100),
        })

    def test_no_drift_identical_data(self, reference_data):
        """Test that identical data shows no drift."""
        detector = DriftDetector(method="psi", threshold=0.2)
        report = detector.detect(reference_data, reference_data, features=["sales"])

        assert isinstance(report, DriftReport)
        assert report.overall_drift_score == pytest.approx(0.0, abs=0.001)
        assert not report.drift_detected

    def test_drift_with_shifted_distribution(self, reference_data):
        """Test that shifted distribution shows drift."""
        current_data = reference_data.copy()
        current_data["sales"] = current_data["sales"] + 50  # Shift mean

        detector = DriftDetector(method="psi", threshold=0.2)
        report = detector.detect(reference_data, current_data, features=["sales"])

        assert report.overall_drift_score > 0.2
        assert report.drift_detected

    def test_multiple_features(self, reference_data):
        """Test drift detection on multiple features."""
        current_data = reference_data.copy()
        current_data["sales"] = current_data["sales"] + 30

        detector = DriftDetector(method="psi")
        report = detector.detect(
            reference_data, current_data, features=["sales", "price"]
        )

        assert len(report.feature_drifts) == 2
        assert "sales" in report.feature_drifts
        assert "price" in report.feature_drifts

    def test_auto_feature_selection(self, reference_data):
        """Test automatic selection of numeric features."""
        detector = DriftDetector(method="psi")
        report = detector.detect(reference_data, reference_data)

        # Should auto-select sales and price (numeric)
        assert len(report.feature_drifts) >= 1

    def test_feature_drift_result_structure(self, reference_data):
        """Test that FeatureDriftResult has correct structure."""
        current_data = reference_data.copy()
        current_data["sales"] = current_data["sales"] + 40

        detector = DriftDetector(method="psi")
        report = detector.detect(reference_data, current_data, features=["sales"])

        result = report.feature_drifts["sales"]
        assert result.feature_name == "sales"
        assert result.metric == "psi"
        assert isinstance(result.statistic, float)
        assert result.p_value is None  # PSI doesn't have p-value
        assert isinstance(result.drift_detected, bool)
        assert "mean" in result.reference_distribution
        assert "mean" in result.current_distribution


class TestKSDetection:
    """Test KS drift detection."""

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset."""
        np.random.seed(42)
        return pd.DataFrame({
            "sales": np.random.normal(100, 15, 100),
        })

    def test_no_drift_identical_data(self, reference_data):
        """Test that identical data passes KS test."""
        detector = DriftDetector(method="ks", threshold=0.05)
        report = detector.detect(reference_data, reference_data, features=["sales"])

        # Same data should have high p-value (no drift)
        assert not report.drift_detected

    def test_drift_with_different_distribution(self, reference_data):
        """Test that different distribution fails KS test."""
        np.random.seed(43)
        current_data = pd.DataFrame({
            "sales": np.random.normal(150, 20, 100),  # Different distribution
        })

        detector = DriftDetector(method="ks", threshold=0.05)
        report = detector.detect(reference_data, current_data, features=["sales"])

        # Different distributions should have low p-value (drift detected)
        assert report.drift_detected

    def test_ks_has_p_value(self, reference_data):
        """Test that KS results include p-value."""
        detector = DriftDetector(method="ks")
        report = detector.detect(reference_data, reference_data, features=["sales"])

        result = report.feature_drifts["sales"]
        assert result.p_value is not None
        assert isinstance(result.p_value, float)


class TestComputePsiSummary:
    """Test detailed PSI computation."""

    def test_psi_summary_structure(self):
        """Test that PSI summary has correct structure."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(100, 15, 100))
        cur = pd.Series(np.random.normal(100, 15, 100))

        summary = compute_psi_summary(ref, cur, n_bins=10)

        assert "psi" in summary
        assert "bins" in summary
        assert len(summary["bins"]) == 10

    def test_bin_details(self):
        """Test that bin details are correct."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(100, 15, 100))
        cur = pd.Series(np.random.normal(100, 15, 100))

        summary = compute_psi_summary(ref, cur, n_bins=5)
        bin_info = summary["bins"][0]

        assert "bin_start" in bin_info
        assert "bin_end" in bin_info
        assert "reference_count" in bin_info
        assert "current_count" in bin_info
        assert "psi" in bin_info


class TestDriftReportMethods:
    """Test DriftReport convenience methods."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample drift report."""
        np.random.seed(42)
        ref = pd.DataFrame({
            "sales": np.random.normal(100, 15, 100),
            "price": np.random.normal(50, 5, 100),
        })
        cur = pd.DataFrame({
            "sales": np.random.normal(140, 15, 100),  # Drift
            "price": np.random.normal(50, 5, 100),    # No drift
        })

        detector = DriftDetector(method="psi", threshold=0.2)
        return detector.detect(ref, cur, features=["sales", "price"])

    def test_summary_method(self, sample_report):
        """Test summary method."""
        summary = sample_report.summary()
        assert isinstance(summary, str)
        assert "features" in summary
        assert str(sample_report.threshold_used) in summary

    def test_get_drifting_features(self, sample_report):
        """Test getting drifting features."""
        drifting = sample_report.get_drifting_features()
        assert isinstance(drifting, list)
        # sales should be drifting, price should not
        assert "sales" in drifting

    def test_report_string_representation(self, sample_report):
        """Test string representation of report."""
        assert str(sample_report.overall_drift_score) in repr(sample_report)
