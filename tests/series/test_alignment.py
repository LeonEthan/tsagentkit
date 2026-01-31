"""Tests for series/alignment.py."""

import pandas as pd
import pytest

from tsagentkit.series import align_timezone, fill_gaps, resample_series


class TestAlignTimezone:
    """Tests for align_timezone function."""

    def test_naive_to_utc(self) -> None:
        """Test converting naive datetime to UTC."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01 00:00:00"]),
            "y": [1.0],
        })
        result = align_timezone(df, target_tz="UTC")
        assert result["ds"].dt.tz is not None
        assert str(result["ds"].dt.tz) == "UTC"

    def test_utc_to_different_tz(self) -> None:
        """Test converting UTC to different timezone."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01 00:00:00"]).tz_localize("UTC"),
            "y": [1.0],
        })
        result = align_timezone(df, target_tz="US/Eastern")
        assert str(result["ds"].dt.tz) == "US/Eastern"

    def test_to_naive(self) -> None:
        """Test converting timezone-aware to naive."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01 00:00:00"]).tz_localize("UTC"),
            "y": [1.0],
        })
        result = align_timezone(df, target_tz=None)
        assert result["ds"].dt.tz is None

    def test_missing_column(self) -> None:
        """Test error when ds column missing."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "y": [1.0],
        })
        with pytest.raises(ValueError, match="not found"):
            align_timezone(df, target_tz="UTC")

    def test_invalid_column_type(self) -> None:
        """Test error when ds column not datetime."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": ["not-a-date"],
            "y": [1.0],
        })
        with pytest.raises(ValueError, match="datetime"):
            align_timezone(df, target_tz="UTC")


class TestResampleSeries:
    """Tests for resample_series function."""

    def test_daily_to_weekly_sum(self) -> None:
        """Test resampling daily to weekly with sum."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 14,
            "ds": pd.date_range("2024-01-01", periods=14, freq="D"),
            "y": [1.0] * 14,
        })
        result = resample_series(df, freq="W", agg_func="sum")
        # Should have 2 weeks of data
        assert len(result) == 2
        assert result["y"].iloc[0] == 7.0  # Sum of 7 days

    def test_hourly_to_daily_mean(self) -> None:
        """Test resampling hourly to daily with mean."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 48,
            "ds": pd.date_range("2024-01-01", periods=48, freq="h"),
            "y": list(range(48)),
        })
        result = resample_series(df, freq="D", agg_func="mean")
        # Should have 2 days
        assert len(result) == 2

    def test_multiple_series_resample(self) -> None:
        """Test resampling with multiple series."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 14 + ["B"] * 14,
            "ds": list(pd.date_range("2024-01-01", periods=14, freq="D")) * 2,
            "y": [1.0] * 14 + [2.0] * 14,
        })
        result = resample_series(df, freq="W", agg_func="sum")
        # Should have 2 weeks * 2 series = 4 rows
        assert len(result) == 4

    def test_agg_func_last(self) -> None:
        """Test resampling with last aggregation."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 7,
            "ds": pd.date_range("2024-01-01", periods=7, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        })
        result = resample_series(df, freq="W", agg_func="last")
        assert result["y"].iloc[0] == 7.0  # Last value of week

    def test_missing_columns(self) -> None:
        """Test error when required columns missing."""
        df = pd.DataFrame({
            "unique_id": ["A"],
        })
        with pytest.raises(ValueError, match="Missing"):
            resample_series(df, freq="D")


class TestFillGaps:
    """Tests for fill_gaps function."""

    def test_fill_with_interpolation(self) -> None:
        """Test gap filling with interpolation."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "y": [1.0, 3.0],
        })
        result = fill_gaps(df, freq="D", method="interpolate")
        # Should have 3 days now
        assert len(result) == 3
        # Middle value should be interpolated to 2.0
        assert result["y"].iloc[1] == 2.0

    def test_fill_with_forward_fill(self) -> None:
        """Test gap filling with forward fill."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "y": [1.0, 3.0],
        })
        result = fill_gaps(df, freq="D", method="forward")
        # Middle value should be 1.0 (forward filled)
        assert result["y"].iloc[1] == 1.0

    def test_fill_with_zero(self) -> None:
        """Test gap filling with zeros."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "y": [1.0, 3.0],
        })
        result = fill_gaps(df, freq="D", method="zero")
        assert result["y"].iloc[1] == 0.0

    def test_no_gaps(self) -> None:
        """Test filling when no gaps exist."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 3,
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        })
        result = fill_gaps(df, freq="D", method="interpolate")
        assert len(result) == 3
        assert list(result["y"]) == [1.0, 2.0, 3.0]

    def test_multiple_series_fill(self) -> None:
        """Test gap filling with multiple series."""
        df = pd.DataFrame({
            "unique_id": ["A", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "y": [1.0, 2.0],
        })
        result = fill_gaps(df, freq="D", method="zero")
        # Each series should have same length
        assert len(result[result["unique_id"] == "A"]) == len(result[result["unique_id"] == "B"])
