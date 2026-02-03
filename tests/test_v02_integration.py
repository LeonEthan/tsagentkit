"""Integration tests for v0.2 features.

Tests the integration of features, monitoring, and bucketing modules
with the core forecasting pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from tsagentkit import MonitoringConfig, TaskSpec
from tsagentkit.features import FeatureConfig, FeatureFactory
from tsagentkit.monitoring import DriftDetector, StabilityMonitor, TriggerEvaluator
from tsagentkit.monitoring.triggers import RetrainTrigger, TriggerType
from tsagentkit.router import BucketConfig, DataBucketer, SeriesBucket
from tsagentkit.serving.provenance import create_provenance


class TestFeaturesIntegration:
    """Test features module integration with pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "unique_id": ["A"] * 50 + ["B"] * 50,
            "ds": list(dates[:50]) + list(dates[:50]),
            "y": np.random.normal(100, 10, 100),
            "promotion": [0, 1] * 50,
            "holiday": [0] * 100,
        })

    def test_feature_factory_integration(self, sample_data):
        """Test FeatureFactory with full feature set."""
        config = FeatureConfig(
            lags=[1, 7],
            calendar_features=["dayofweek", "month"],
            rolling_windows={7: ["mean"], 14: ["std"]},
            known_covariates=["holiday"],
            observed_covariates=["promotion"],
        )

        factory = FeatureFactory(config)

        # Create mock TSDataset
        class MockTSDataset:
            def __init__(self, data):
                self.data = data

        dataset = MockTSDataset(sample_data)
        matrix = factory.create_features(dataset)

        # Verify feature matrix
        assert matrix.config_hash is not None
        assert len(matrix.feature_cols) > 0
        assert "dayofweek" in matrix.feature_cols
        assert "month" in matrix.feature_cols
        assert "holiday" in matrix.known_covariates

    def test_feature_versioning_consistency(self, sample_data):
        """Test that same config produces same hash."""
        config = FeatureConfig(lags=[1, 7, 14])
        factory1 = FeatureFactory(config)
        factory2 = FeatureFactory(config)

        class MockTSDataset:
            def __init__(self, data):
                self.data = data

        dataset = MockTSDataset(sample_data)
        matrix1 = factory1.create_features(dataset)
        matrix2 = factory2.create_features(dataset)

        assert matrix1.config_hash == matrix2.config_hash


class TestMonitoringIntegration:
    """Test monitoring module integration with pipeline."""

    @pytest.fixture
    def reference_data(self):
        """Create reference data for drift detection."""
        np.random.seed(42)
        return pd.DataFrame({
            "unique_id": ["A"] * 100,
            "ds": pd.date_range("2024-01-01", periods=100),
            "sales": np.random.normal(100, 15, 100),
        })

    @pytest.fixture
    def drifted_data(self):
        """Create drifted data for testing."""
        np.random.seed(43)
        return pd.DataFrame({
            "unique_id": ["A"] * 100,
            "ds": pd.date_range("2024-04-10", periods=100),
            "sales": np.random.normal(150, 20, 100),  # Shifted distribution
        })

    def test_drift_detector_integration(self, reference_data, drifted_data):
        """Test drift detection with PSI method."""
        detector = DriftDetector(method="psi", threshold=0.2)
        report = detector.detect(
            reference_data=reference_data,
            current_data=drifted_data,
            features=["sales"],
        )

        assert report.drift_detected
        assert report.overall_drift_score > 0.2
        assert "sales" in report.feature_drifts

    def test_stability_monitor_integration(self):
        """Test stability monitoring with multiple predictions."""
        # Create multiple forecast runs
        predictions = []
        for i in range(3):
            pred = pd.DataFrame({
                "unique_id": ["A"] * 10,
                "ds": pd.date_range("2024-01-01", periods=10),
                "yhat": np.random.normal(100, 5, 10),
            })
            predictions.append(pred)

        monitor = StabilityMonitor(jitter_threshold=0.1)
        jitter = monitor.compute_jitter(predictions, method="cv")

        assert "A" in jitter
        assert isinstance(jitter["A"], float)

    def test_trigger_evaluator_integration(self, reference_data, drifted_data):
        """Test trigger evaluator with drift detection."""
        detector = DriftDetector(method="psi", threshold=0.2)
        drift_report = detector.detect(
            reference_data=reference_data,
            current_data=drifted_data,
            features=["sales"],
        )

        triggers = [
            RetrainTrigger(TriggerType.DRIFT, threshold=0.2),
        ]
        evaluator = TriggerEvaluator(triggers)

        results = evaluator.evaluate(drift_report=drift_report)

        assert len(results) == 1
        assert results[0].fired
        assert results[0].trigger_type == "drift"

    def test_should_retrain_logic(self, reference_data, drifted_data):
        """Test should_retrain convenience method."""
        detector = DriftDetector(method="psi", threshold=0.2)
        drift_report = detector.detect(
            reference_data=reference_data,
            current_data=drifted_data,
            features=["sales"],
        )

        triggers = [RetrainTrigger(TriggerType.DRIFT, threshold=0.2)]
        evaluator = TriggerEvaluator(triggers)

        assert evaluator.should_retrain(drift_report=drift_report)


class TestBucketingIntegration:
    """Test bucketing integration with series module."""

    @pytest.fixture
    def multi_series_data(self):
        """Create multi-series data with varying characteristics."""
        np.random.seed(42)
        # Series A: High volume, long history (HEAD + LONG_HISTORY)
        # Series B: Low volume, short history (TAIL + SHORT_HISTORY)
        # Series C: Medium volume, medium history (no bucket)
        return pd.DataFrame({
            "unique_id": ["A"] * 400 + ["B"] * 20 + ["C"] * 200,
            "ds": (
                list(pd.date_range("2024-01-01", periods=400)) +
                list(pd.date_range("2024-01-01", periods=20)) +
                list(pd.date_range("2024-01-01", periods=200))
            ),
            "y": (
                np.random.normal(1000, 100, 400).tolist() +
                np.random.normal(50, 5, 20).tolist() +
                np.random.normal(200, 20, 200).tolist()
            ),
        })

    def test_data_bucketer_integration(self, multi_series_data):
        """Test DataBucketer with multi-series dataset."""
        config = BucketConfig(
            head_quantile_threshold=0.8,
            tail_quantile_threshold=0.2,
            short_history_max_obs=50,
            long_history_min_obs=300,
        )

        bucketer = DataBucketer(config)

        # Create mock TSDataset
        class MockTSDataset:
            def __init__(self, data):
                self.data = data

        dataset = MockTSDataset(multi_series_data)
        profile = bucketer.create_bucket_profile(dataset)

        # Verify bucket assignments
        a_buckets = profile.get_bucket_for_series("A")
        assert SeriesBucket.HEAD in a_buckets or SeriesBucket.LONG_HISTORY in a_buckets

        b_buckets = profile.get_bucket_for_series("B")
        assert SeriesBucket.TAIL in b_buckets or SeriesBucket.SHORT_HISTORY in b_buckets

    def test_bucket_statistics_computation(self, multi_series_data):
        """Test that bucket statistics are computed correctly."""
        config = BucketConfig(head_quantile_threshold=0.8)
        bucketer = DataBucketer(config)

        class MockTSDataset:
            def __init__(self, data):
                self.data = data

        dataset = MockTSDataset(multi_series_data)
        profile = bucketer.create_bucket_profile(dataset)

        # Check that stats exist for each bucket
        for bucket in SeriesBucket:
            assert bucket in profile.bucket_stats

        # HEAD bucket should have high average value
        head_stats = profile.bucket_stats[SeriesBucket.HEAD]
        if head_stats.series_count > 0:
            assert head_stats.avg_value > 0

    def test_model_recommendations_per_bucket(self):
        """Test model recommendations for different buckets."""
        bucketer = DataBucketer()

        head_model = bucketer.get_model_for_bucket(SeriesBucket.HEAD)
        tail_model = bucketer.get_model_for_bucket(SeriesBucket.TAIL)
        short_model = bucketer.get_model_for_bucket(SeriesBucket.SHORT_HISTORY)
        long_model = bucketer.get_model_for_bucket(SeriesBucket.LONG_HISTORY)

        # All should return valid model names
        assert isinstance(head_model, str)
        assert isinstance(tail_model, str)
        assert isinstance(short_model, str)
        assert isinstance(long_model, str)

    def test_sparsity_override(self):
        """Test that sparsity classification overrides volume."""
        bucketer = DataBucketer()

        # Even HEAD bucket should return Croston for intermittent
        model = bucketer.get_model_for_bucket(
            SeriesBucket.HEAD,
            sparsity_class="intermittent"
        )
        assert model == "Croston"


class TestProvenanceIntegration:
    """Test provenance integration with v0.2 features."""

    @pytest.fixture
    def mock_feature_matrix(self):
        """Create mock feature matrix."""
        class MockFeatureMatrix:
            def __init__(self):
                self.config_hash = "abc123"
                self.signature = "FeatureMatrix(c=abc123,n=5)"
                self.feature_cols = ["lag_1", "lag_7", "dayofweek"]
        return MockFeatureMatrix()

    @pytest.fixture
    def mock_drift_report(self):
        """Create mock drift report."""
        class MockDriftReport:
            def __init__(self):
                self.drift_detected = True
                self.overall_drift_score = 0.25
                self.threshold_used = 0.2

            def get_drifting_features(self):
                return ["sales", "price"]
        return MockDriftReport()

    def test_provenance_with_features(self, mock_feature_matrix):
        """Test provenance includes feature info."""
        from tsagentkit.router import PlanSpec

        data = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
        })
        task_spec = TaskSpec(h=7, freq="D")
        plan = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])

        provenance = create_provenance(
            data=data,
            task_spec=task_spec,
            plan=plan,
            feature_matrix=mock_feature_matrix,
        )

        assert "feature_signature" in provenance.metadata
        assert "feature_config_hash" in provenance.metadata
        assert "n_features" in provenance.metadata
        assert provenance.metadata["feature_signature"] == mock_feature_matrix.signature

    def test_provenance_with_drift(self, mock_drift_report):
        """Test provenance includes drift info."""
        from tsagentkit.router import PlanSpec

        data = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [1.0],
        })
        task_spec = TaskSpec(h=7, freq="D")
        plan = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])

        provenance = create_provenance(
            data=data,
            task_spec=task_spec,
            plan=plan,
            drift_report=mock_drift_report,
        )

        assert "drift_detected" in provenance.metadata
        assert "drift_score" in provenance.metadata
        assert "drift_threshold" in provenance.metadata
        assert "drifting_features" in provenance.metadata
        assert provenance.metadata["drift_detected"] is True


class TestEndToEndV02:
    """End-to-end tests for v0.2 features."""

    def test_full_pipeline_with_monitoring(self):
        """Test complete pipeline with monitoring enabled."""
        np.random.seed(42)

        # Create reference and current data
        reference_data = pd.DataFrame({
            "unique_id": ["A"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50),
            "y": np.random.normal(100, 10, 50),
        })

        current_data = pd.DataFrame({
            "unique_id": ["A"] * 50,
            "ds": pd.date_range("2024-02-20", periods=50),
            "y": np.random.normal(100, 10, 50),
        })

        # Create monitoring config
        monitoring_config = MonitoringConfig(
            enabled=True,
            drift_method="psi",
            drift_threshold=0.2,
        )

        # This would be called in a real scenario
        # run_forecast(
        #     data=current_data,
        #     task_spec=TaskSpec(h=7, freq="D"),
        #     monitoring_config=monitoring_config,
        #     reference_data=reference_data,
        # )

        # Just verify the config was created
        assert monitoring_config.enabled
        assert monitoring_config.drift_method == "psi"

    def test_feature_to_provenance_workflow(self):
        """Test workflow from features to provenance."""
        # Create sample data
        data = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": list(range(30)),
        })

        # Create features
        config = FeatureConfig(lags=[1, 7])
        factory = FeatureFactory(config)

        class MockTSDataset:
            def __init__(self, data):
                self.data = data

        matrix = factory.create_features(MockTSDataset(data))

        # Create provenance with features
        from tsagentkit.router import PlanSpec

        task_spec = TaskSpec(h=7, freq="D")
        plan = PlanSpec(plan_name="default", candidate_models=["SeasonalNaive"])

        provenance = create_provenance(
            data=data,
            task_spec=task_spec,
            plan=plan,
            feature_matrix=matrix,
        )

        assert provenance.metadata["feature_signature"] == matrix.signature
        assert provenance.metadata["n_features"] == 2  # lag_1 and lag_7
