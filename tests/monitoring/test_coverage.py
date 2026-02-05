"""Tests for coverage monitoring."""

from __future__ import annotations

import pandas as pd

from tsagentkit.monitoring import CoverageMonitor


class TestCoverageMonitor:
    """Test coverage monitoring functionality."""

    def test_check_interval_coverage(self) -> None:
        """Test checking interval coverage."""
        # Create forecasts with quantile columns
        forecasts = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B", "B"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06"] * 2),
            "q_0.1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "q_0.9": [9.0, 10.0, 11.0, 9.0, 10.0, 11.0],
        })

        # Create actuals (all within 80% interval)
        actuals = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B", "B"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06"] * 2),
            "y": [5.0, 6.0, 7.0, 5.0, 6.0, 7.0],
        })

        monitor = CoverageMonitor()
        checks = monitor.check(forecasts, actuals, quantiles=[0.1, 0.9])

        assert len(checks) == 1
        assert checks[0].expected_coverage == 0.8
        assert checks[0].actual_coverage == 1.0  # All within interval
        assert checks[0].is_acceptable()

    def test_check_coverage_outside_interval(self) -> None:
        """Test coverage when values are outside interval."""
        forecasts = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06"]),
            "q_0.1": [2.0, 3.0, 4.0],
            "q_0.9": [4.0, 5.0, 6.0],
        })

        # Values outside interval - y=1 is below q_0.1, y=10 is above q_0.9
        actuals = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06"]),
            "y": [1.0, 3.5, 10.0],  # y=3.5 is within [3.0, 5.0], others outside
        })

        monitor = CoverageMonitor()
        checks = monitor.check(forecasts, actuals, quantiles=[0.1, 0.9])

        assert len(checks) == 1
        # Only y=3.5 is within [3.0, 5.0], so coverage is 1/3
        assert checks[0].actual_coverage == 1 / 3
        assert not checks[0].is_acceptable()

    def test_check_single_quantile(self) -> None:
        """Test checking single quantile coverage."""
        forecasts = pd.DataFrame({
            "unique_id": ["A", "A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]),
            "q_0.5": [5.0, 6.0, 7.0, 8.0],
        })

        # 2 out of 4 below or equal to median (y <= q_0.5)
        # y=4.0 <= 5.0: True
        # y=5.5 <= 6.0: True
        # y=6.5 <= 7.0: True
        # y=10.0 <= 8.0: False
        # So 3 out of 4 are below/equal -> 0.75 coverage
        actuals = pd.DataFrame({
            "unique_id": ["A", "A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]),
            "y": [4.0, 5.5, 6.5, 10.0],
        })

        monitor = CoverageMonitor()
        check = monitor.check_single_quantile(forecasts, actuals, quantile=0.5)

        assert check is not None
        assert check.expected_coverage == 0.5
        # 3 out of 4 are <= q_0.5
        assert check.actual_coverage == 0.75
        assert check.is_acceptable()  # 0.75 >= 0.5 - 0.05 = 0.45

    def test_check_with_horizon(self) -> None:
        """Test coverage check with horizon column."""
        forecasts = pd.DataFrame({
            "unique_id": ["A", "A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]),
            "h": [1, 2, 1, 2],
            "q_0.1": [1.0, 2.0, 1.0, 2.0],
            "q_0.9": [9.0, 10.0, 9.0, 10.0],
        })

        actuals = pd.DataFrame({
            "unique_id": ["A", "A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]),
            "y": [5.0, 6.0, 5.0, 6.0],
        })

        monitor = CoverageMonitor()
        checks = monitor.check(forecasts, actuals, quantiles=[0.1, 0.9])

        assert len(checks) == 1
        assert 1 in checks[0].hit_rate_by_horizon
        assert 2 in checks[0].hit_rate_by_horizon

    def test_empty_data(self) -> None:
        """Test with empty data."""
        forecasts = pd.DataFrame()
        actuals = pd.DataFrame()

        monitor = CoverageMonitor()
        checks = monitor.check(forecasts, actuals, quantiles=[0.1, 0.9])

        assert len(checks) == 0

    def test_missing_columns(self) -> None:
        """Test with missing required columns."""
        forecasts = pd.DataFrame({
            "wrong_col": [1, 2, 3],
        })
        actuals = pd.DataFrame({
            "wrong_col": [1, 2, 3],
        })

        monitor = CoverageMonitor()
        checks = monitor.check(forecasts, actuals, quantiles=[0.1, 0.9])

        assert len(checks) == 0
