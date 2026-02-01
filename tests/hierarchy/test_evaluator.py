"""Tests for hierarchy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit.hierarchy import (
    CoherenceViolation,
    HierarchyEvaluationReport,
    HierarchyEvaluator,
    HierarchyStructure,
)


@pytest.fixture
def simple_structure():
    """Create a simple 2-level hierarchy."""
    # all_nodes = ["A", "B", "Total"]
    return HierarchyStructure(
        aggregation_graph={
            "Total": ["A", "B"],
        },
        bottom_nodes=["A", "B"],
        s_matrix=np.array([
            #A  B
            [1, 0],  # A (index 0)
            [0, 1],  # B (index 1)
            [1, 1],  # Total (index 2)
        ]),
    )


@pytest.fixture
def coherent_forecasts():
    """Create coherent forecast DataFrame."""
    return pd.DataFrame({
        "unique_id": ["Total", "A", "B"] * 3,
        "ds": ["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
        "yhat": [10, 4, 6, 12, 5, 7, 14, 6, 8],  # Total = A + B
    })


@pytest.fixture
def incoherent_forecasts():
    """Create incoherent forecast DataFrame."""
    return pd.DataFrame({
        "unique_id": ["Total", "A", "B"] * 3,
        "ds": ["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
        "yhat": [15, 4, 6, 20, 5, 7, 25, 6, 8],  # Total != A + B
    })


class TestCoherenceViolation:
    """Test CoherenceViolation dataclass."""

    def test_creation(self):
        """Test creating a coherence violation."""
        violation = CoherenceViolation(
            parent_node="Total",
            child_nodes=["A", "B"],
            expected_value=10.0,
            actual_value=15.0,
            difference=5.0,
            timestamp="2024-01-01",
        )

        assert violation.parent_node == "Total"
        assert violation.child_nodes == ["A", "B"]
        assert violation.expected_value == 10.0
        assert violation.actual_value == 15.0
        assert violation.difference == 5.0
        assert violation.timestamp == "2024-01-01"


class TestHierarchyEvaluationReport:
    """Test HierarchyEvaluationReport dataclass."""

    def test_creation(self):
        """Test creating an evaluation report."""
        report = HierarchyEvaluationReport(
            coherence_score=0.95,
            total_violations=2,
            violation_rate=0.1,
        )

        assert report.coherence_score == 0.95
        assert report.total_violations == 2
        assert report.violation_rate == 0.1

    def test_to_dict(self):
        """Test converting report to dictionary."""
        violations = [
            CoherenceViolation(
                parent_node="Total",
                child_nodes=["A", "B"],
                expected_value=10.0,
                actual_value=15.0,
                difference=5.0,
                timestamp="2024-01-01",
            )
        ]

        report = HierarchyEvaluationReport(
            coherence_score=0.95,
            coherence_violations=violations,
        )

        d = report.to_dict()
        assert d["coherence_score"] == 0.95
        assert len(d["coherence_violations"]) == 1
        assert d["coherence_violations"][0]["parent_node"] == "Total"


class TestHierarchyEvaluator:
    """Test HierarchyEvaluator class."""

    def test_initialization(self, simple_structure):
        """Test evaluator initialization."""
        evaluator = HierarchyEvaluator(simple_structure)
        assert evaluator.structure == simple_structure

    def test_evaluate_coherent_forecasts(self, simple_structure, coherent_forecasts):
        """Test evaluating coherent forecasts."""
        evaluator = HierarchyEvaluator(simple_structure)
        report = evaluator.evaluate(coherent_forecasts)

        assert report.coherence_score == 1.0
        assert report.total_violations == 0
        assert report.violation_rate == 0.0

    def test_evaluate_incoherent_forecasts(self, simple_structure, incoherent_forecasts):
        """Test evaluating incoherent forecasts."""
        evaluator = HierarchyEvaluator(simple_structure)
        report = evaluator.evaluate(incoherent_forecasts)

        assert report.coherence_score < 1.0
        assert report.total_violations > 0
        assert report.violation_rate > 0.0

    def test_detect_violations(self, simple_structure, incoherent_forecasts):
        """Test violation detection."""
        evaluator = HierarchyEvaluator(simple_structure)
        violations = evaluator._detect_violations(incoherent_forecasts, tolerance=1e-6)

        assert len(violations) == 3  # One per timestamp

        # Check first violation
        v = violations[0]
        assert v.parent_node == "Total"
        assert v.child_nodes == ["A", "B"]
        assert v.expected_value == 10.0  # A + B = 4 + 6
        assert v.actual_value == 15.0    # Total = 15
        assert v.difference == 5.0

    def test_coherence_score_calculation(self, simple_structure):
        """Test coherence score calculation."""
        evaluator = HierarchyEvaluator(simple_structure)

        # Perfectly coherent
        coherent = pd.DataFrame({
            "unique_id": ["Total", "A", "B"],
            "ds": ["2024-01-01"] * 3,
            "yhat": [10, 4, 6],
        })
        score = evaluator._compute_coherence_score(coherent)
        assert score == 1.0

        # Incoherent
        incoherent = pd.DataFrame({
            "unique_id": ["Total", "A", "B"],
            "ds": ["2024-01-01"] * 3,
            "yhat": [15, 4, 6],  # Total=15, A+B=10, violation=5
        })
        score = evaluator._compute_coherence_score(incoherent)
        assert score < 1.0
        # Total sum = 15+4+6 = 25, violation = 5
        # coherence_score = 1 - (5/25) = 0.8
        # But since Total has the violation: 1 - (5/15) = 0.67
        assert abs(score - 0.667) < 0.01

    def test_compute_level_metrics(self, simple_structure):
        """Test computing metrics by hierarchy level."""
        evaluator = HierarchyEvaluator(simple_structure)

        forecasts = pd.DataFrame({
            "unique_id": ["Total", "A", "B"] * 2,
            "ds": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "yhat": [10, 4, 6, 12, 5, 7],
        })

        actuals = pd.DataFrame({
            "unique_id": ["Total", "A", "B"] * 2,
            "ds": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "y": [11, 5, 6, 13, 6, 7],
        })

        metrics = evaluator._compute_level_metrics(forecasts, actuals)

        # Should have metrics for levels 0 and 1
        assert 0 in metrics
        assert 1 in metrics

        # Level 0 (Total) should have metrics
        assert "mae" in metrics[0]
        assert metrics[0]["count"] == 2  # 2 time points

    def test_compute_improvement(self, simple_structure):
        """Test computing improvement metrics."""
        evaluator = HierarchyEvaluator(simple_structure)

        base = pd.DataFrame({
            "unique_id": ["Total", "A", "B"],
            "ds": ["2024-01-01"] * 3,
            "yhat": [10, 4, 6],
        })

        reconciled = pd.DataFrame({
            "unique_id": ["Total", "A", "B"],
            "ds": ["2024-01-01"] * 3,
            "yhat": [11, 5, 6],
        })

        actuals = pd.DataFrame({
            "unique_id": ["Total", "A", "B"],
            "ds": ["2024-01-01"] * 3,
            "y": [12, 6, 6],
        })

        improvement = evaluator.compute_improvement(base, reconciled, actuals)

        # MAE improved from |10-12|+|4-6|+|6-6|=4 to |11-12|+|5-6|+|6-6|=2
        assert improvement["mae"] > 0  # Percentage improvement

    def test_tolerance_parameter(self, simple_structure):
        """Test that tolerance affects violation detection."""
        evaluator = HierarchyEvaluator(simple_structure)

        # Small violation (within tolerance)
        forecasts = pd.DataFrame({
            "unique_id": ["Total", "A", "B"],
            "ds": ["2024-01-01"] * 3,
            "yhat": [10.0001, 4, 6],  # Very small violation
        })

        violations_strict = evaluator._detect_violations(forecasts, tolerance=1e-10)
        violations_loose = evaluator._detect_violations(forecasts, tolerance=1e-3)

        assert len(violations_strict) == 1
        assert len(violations_loose) == 0
