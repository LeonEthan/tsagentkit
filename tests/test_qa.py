"""Tests for QA module.

Tests quality assurance checks and reports.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit.qa import QAReport, run_qa


@pytest.fixture
def valid_df():
    """Create valid DataFrame."""
    return pd.DataFrame({
        "unique_id": ["A"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50),
        "y": range(50),
    })


class TestQAReport:
    """Test QAReport dataclass."""

    def test_empty_report(self):
        """Create empty QA report."""
        report = QAReport()
        assert report.issues == []
        assert report.valid is True
        assert not report.has_critical_issues()

    def test_report_with_issues(self):
        """Create report with issues."""
        issues = [
            {"type": "test", "severity": "warning"},
            {"type": "test2", "severity": "critical"},
        ]
        report = QAReport(issues=issues, valid=False)
        assert len(report.issues) == 2
        assert report.has_critical_issues()

    def test_report_valid_without_critical(self):
        """Report is valid without critical issues."""
        issues = [
            {"type": "test", "severity": "warning"},
        ]
        report = QAReport(issues=issues, valid=True)
        assert not report.has_critical_issues()


class TestRunQA:
    """Test run_qa function."""

    def test_valid_data(self, valid_df):
        """QA on valid data returns valid report."""
        report = run_qa(valid_df, min_train_size=30)  # Lower min_train_size
        assert report.valid is True
        assert len(report.issues) == 0

    def test_short_series(self):
        """Short series generates warning."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "y": range(10),
        })
        report = run_qa(df, min_train_size=20)
        # min_history is a warning, not critical, so report is still valid
        assert report.valid  # Warnings don't invalidate
        assert any(issue["type"] == "min_history" for issue in report.issues)

    def test_duplicates_detected(self):
        """Duplicates are detected as critical."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "y": [1, 2, 3],
        })
        report = run_qa(df)
        assert not report.valid
        assert report.has_critical_issues()
        assert any(issue["type"] == "duplicates" for issue in report.issues)

    def test_not_monotonic(self):
        """Non-monotonic time series is critical."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04", "2024-01-05"]),
            "y": range(5),
        })
        report = run_qa(df)
        assert not report.valid
        assert report.has_critical_issues()
        assert any(issue["type"] == "not_monotonic" for issue in report.issues)

    def test_multiple_series_all_valid(self):
        """Multiple valid series."""
        dfs = []
        for uid in ["A", "B", "C"]:
            df = pd.DataFrame({
                "unique_id": [uid] * 30,
                "ds": pd.date_range("2024-01-01", periods=30),
                "y": range(30),
            })
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        report = run_qa(df)
        assert report.valid is True

    def test_multiple_series_one_short(self):
        """Multiple series with one short."""
        dfs = []
        for uid in ["A", "B", "C"]:
            length = 20 if uid == "A" else 60
            df = pd.DataFrame({
                "unique_id": [uid] * length,
                "ds": pd.date_range("2024-01-01", periods=length),
                "y": range(length),
            })
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        report = run_qa(df, min_train_size=30)
        # min_history is a warning, not critical
        assert report.valid  # Warnings don't invalidate
        assert any(issue["type"] == "min_history" for issue in report.issues)

    def test_multiple_series_one_unsorted(self):
        """Multiple series with one unsorted."""
        df_a = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "y": range(5),
        })
        df_b = pd.DataFrame({
            "unique_id": ["B"] * 5,
            "ds": pd.to_datetime(["2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01"]),
            "y": range(5),
        })
        df = pd.concat([df_a, df_b], ignore_index=True)

        report = run_qa(df)
        assert not report.valid
        assert any(issue["type"] == "not_monotonic" and issue["series"] == "B" for issue in report.issues)

    def test_custom_columns(self):
        """QA with custom column names."""
        df = pd.DataFrame({
            "series_id": ["A"] * 20,
            "timestamp": pd.date_range("2024-01-01", periods=20),
            "value": range(20),
        })
        report = run_qa(df, id_col="series_id", time_col="timestamp", target_col="value")
        assert report.valid is True

    def test_duplicate_count(self):
        """Duplicate count is accurate."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]),
            "y": [1, 2, 3, 4, 5],
        })
        report = run_qa(df)
        dup_issue = next(issue for issue in report.issues if issue["type"] == "duplicates")
        # Duplicate count: 2 for A-01, 2 for B-01 = 4 duplicate rows total
        assert dup_issue["count"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
