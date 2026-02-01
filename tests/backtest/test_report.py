"""Tests for backtest/report.py."""

import pytest

from tsagentkit.backtest.report import (
    BacktestReport,
    SegmentMetrics,
    SeriesMetrics,
    TemporalMetrics,
    WindowResult,
)


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


class TestSegmentMetrics:
    """Tests for SegmentMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating SegmentMetrics."""
        sm = SegmentMetrics(
            segment_name="intermittent",
            series_ids=["A", "B"],
            metrics={"wape": 0.15, "smape": 0.18},
            n_series=2,
        )
        assert sm.segment_name == "intermittent"
        assert sm.n_series == 2
        assert "A" in sm.series_ids

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        sm = SegmentMetrics(
            segment_name="regular",
            series_ids=["C"],
            metrics={"wape": 0.1},
            n_series=1,
        )
        d = sm.to_dict()
        assert d["segment_name"] == "regular"
        assert d["n_series"] == 1


class TestTemporalMetrics:
    """Tests for TemporalMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating TemporalMetrics."""
        tm = TemporalMetrics(
            dimension="hour",
            metrics_by_value={
                "0": {"wape": 0.12},
                "12": {"wape": 0.15},
            },
        )
        assert tm.dimension == "hour"
        assert "0" in tm.metrics_by_value

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tm = TemporalMetrics(
            dimension="dayofweek",
            metrics_by_value={"Mon": {"wape": 0.1}},
        )
        d = tm.to_dict()
        assert d["dimension"] == "dayofweek"


class TestBacktestReportSegmentAndTemporal:
    """Tests for BacktestReport segment and temporal features."""

    @pytest.fixture
    def sample_report_with_segments(self) -> BacktestReport:
        """Create a BacktestReport with segment and temporal metrics."""
        return BacktestReport(
            n_windows=3,
            strategy="expanding",
            window_results=[],
            aggregate_metrics={"wape": 0.1},
            series_metrics={
                "A": SeriesMetrics(series_id="A", metrics={"wape": 0.08}),
                "B": SeriesMetrics(series_id="B", metrics={"wape": 0.12}),
            },
            segment_metrics={
                "regular": SegmentMetrics(
                    segment_name="regular",
                    series_ids=["A"],
                    metrics={"wape": 0.08},
                    n_series=1,
                ),
                "intermittent": SegmentMetrics(
                    segment_name="intermittent",
                    series_ids=["B"],
                    metrics={"wape": 0.12},
                    n_series=1,
                ),
            },
            temporal_metrics={
                "hour": TemporalMetrics(
                    dimension="hour",
                    metrics_by_value={"0": {"wape": 0.1}, "12": {"wape": 0.15}},
                ),
            },
        )

    def test_get_segment_metric(self, sample_report_with_segments: BacktestReport) -> None:
        """Test getting a metric for a segment."""
        assert sample_report_with_segments.get_segment_metric("regular", "wape") == 0.08
        assert sample_report_with_segments.get_segment_metric("intermittent", "wape") == 0.12

    def test_get_segment_metric_not_found(self, sample_report_with_segments: BacktestReport) -> None:
        """Test getting a metric for a non-existent segment."""
        assert sample_report_with_segments.get_segment_metric("unknown", "wape") != sample_report_with_segments.get_segment_metric("unknown", "wape")  # NaN

    def test_get_temporal_metric(self, sample_report_with_segments: BacktestReport) -> None:
        """Test getting a temporal metric."""
        assert sample_report_with_segments.get_temporal_metric("hour", "0", "wape") == 0.1
        assert sample_report_with_segments.get_temporal_metric("hour", "12", "wape") == 0.15

    def test_compare_segments(self, sample_report_with_segments: BacktestReport) -> None:
        """Test comparing metrics across segments."""
        comparison = sample_report_with_segments.compare_segments("wape")
        assert comparison["regular"] == 0.08
        assert comparison["intermittent"] == 0.12

    def test_summary_with_segments(self, sample_report_with_segments: BacktestReport) -> None:
        """Test summary includes segment information."""
        summary = sample_report_with_segments.summary()
        assert "regular" in summary
        assert "intermittent" in summary
        assert "hour" in summary
