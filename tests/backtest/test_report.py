"""Tests for backtest/report.py."""

import pytest

from tsagentkit.backtest.report import BacktestReport, SeriesMetrics, WindowResult


class TestWindowResult:
    """Tests for WindowResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a WindowResult."""
        result = WindowResult(
            window_index=0,
            train_start="2024-01-01",
            train_end="2024-01-20",
            test_start="2024-01-21",
            test_end="2024-01-27",
            metrics={"wape": 0.1},
            num_series=2,
            num_observations=14,
        )
        assert result.window_index == 0
        assert result.metrics["wape"] == 0.1

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = WindowResult(
            window_index=0,
            train_start="2024-01-01",
            train_end="2024-01-20",
            test_start="2024-01-21",
            test_end="2024-01-27",
            metrics={"wape": 0.1},
        )
        d = result.to_dict()
        assert d["window_index"] == 0
        assert d["metrics"]["wape"] == 0.1


class TestSeriesMetrics:
    """Tests for SeriesMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating SeriesMetrics."""
        sm = SeriesMetrics(
            series_id="A",
            metrics={"wape": 0.1, "smape": 0.15},
            num_windows=5,
        )
        assert sm.series_id == "A"
        assert sm.num_windows == 5


class TestBacktestReport:
    """Tests for BacktestReport dataclass."""

    @pytest.fixture
    def sample_report(self) -> BacktestReport:
        """Create a sample BacktestReport."""
        return BacktestReport(
            n_windows=3,
            strategy="expanding",
            window_results=[
                WindowResult(
                    window_index=0,
                    train_start="2024-01-01",
                    train_end="2024-01-20",
                    test_start="2024-01-21",
                    test_end="2024-01-27",
                    num_series=2,
                    num_observations=14,
                ),
            ],
            aggregate_metrics={"wape": 0.1, "smape": 0.12, "mae": 0.5},
            series_metrics={
                "A": SeriesMetrics(
                    series_id="A",
                    metrics={"wape": 0.08, "smape": 0.10},
                    num_windows=3,
                ),
                "B": SeriesMetrics(
                    series_id="B",
                    metrics={"wape": 0.12, "smape": 0.14},
                    num_windows=3,
                ),
            },
        )

    def test_get_metric(self, sample_report: BacktestReport) -> None:
        """Test getting a metric by name."""
        assert sample_report.get_metric("wape") == 0.1
        assert sample_report.get_metric("smape") == 0.12

    def test_get_metric_not_found(self, sample_report: BacktestReport) -> None:
        """Test getting a non-existent metric."""
        assert sample_report.get_metric("nonexistent") != sample_report.get_metric("nonexistent")  # NaN

    def test_get_series_metric(self, sample_report: BacktestReport) -> None:
        """Test getting a metric for a specific series."""
        assert sample_report.get_series_metric("A", "wape") == 0.08
        assert sample_report.get_series_metric("B", "wape") == 0.12

    def test_get_best_series(self, sample_report: BacktestReport) -> None:
        """Test getting the best series by metric."""
        best = sample_report.get_best_series("wape")
        assert best == "A"  # A has lower WAPE

    def test_get_worst_series(self, sample_report: BacktestReport) -> None:
        """Test getting the worst series by metric."""
        worst = sample_report.get_worst_series("wape")
        assert worst == "B"  # B has higher WAPE

    def test_to_dict(self, sample_report: BacktestReport) -> None:
        """Test conversion to dictionary."""
        d = sample_report.to_dict()
        assert d["n_windows"] == 3
        assert d["strategy"] == "expanding"
        assert "aggregate_metrics" in d
        assert "series_metrics" in d

    def test_summary(self, sample_report: BacktestReport) -> None:
        """Test summary generation."""
        summary = sample_report.summary()
        assert "3 windows" in summary
        assert "expanding" in summary
        assert "wape" in summary
