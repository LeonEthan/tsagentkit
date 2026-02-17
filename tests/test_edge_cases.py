"""Tests for edge cases and boundary conditions.

Tests handling of unusual inputs and extreme scenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, validate
from tsagentkit.core.errors import EContract


class TestEmptyData:
    """Test handling of empty data."""

    def test_empty_dataframe_raises(self):
        """Empty DataFrame raises EContract."""
        df = pd.DataFrame(columns=["unique_id", "ds", "y"])
        with pytest.raises(EContract, match="empty"):
            validate(df)

    def test_no_rows_raises(self):
        """DataFrame with no rows raises."""
        df = pd.DataFrame({
            "unique_id": pd.Series([], dtype=str),
            "ds": pd.Series([], dtype="datetime64[ns]"),
            "y": pd.Series([], dtype=float),
        })
        with pytest.raises(EContract, match="empty"):
            validate(df)


class TestNullValues:
    """Test handling of null values."""

    def test_null_in_target_raises(self):
        """Null in target column raises."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": list(range(15)) + [None] + list(range(15, 29)),
        })
        with pytest.raises(EContract, match="null"):
            validate(df)

    def test_null_in_id_raises(self):
        """Null in unique_id column raises."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 15 + [None] + ["A"] * 14,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": range(30),
        })
        with pytest.raises(EContract, match="null"):
            validate(df)

    def test_null_in_ds_raises(self):
        """Null in ds column raises."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": list(pd.date_range("2024-01-01", periods=15)) + [None] + list(pd.date_range("2024-01-16", periods=14)),
            "y": range(30),
        })
        with pytest.raises(EContract, match="null"):
            validate(df)


class TestExtremeHorizons:
    """Test extreme forecast horizons."""

    def test_h_zero_raises(self):
        """h=0 raises ValueError."""
        with pytest.raises(ValueError, match="h must be positive"):
            ForecastConfig(h=0, freq="D")

    def test_negative_h_raises(self):
        """Negative h raises ValueError."""
        with pytest.raises(ValueError, match="h must be positive"):
            ForecastConfig(h=-5, freq="D")

    def test_very_large_h(self):
        """Very large h is allowed (may cause memory issues but should not crash)."""
        config = ForecastConfig(h=1000, freq="D")
        assert config.h == 1000


class TestExtremeValues:
    """Test extreme values in data."""

    def test_very_large_values(self):
        """Very large values are handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": [1e10] * 30,
        })
        result = validate(df)
        assert len(result) == 30

    def test_very_small_values(self):
        """Very small values are handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": [1e-10] * 30,
        })
        result = validate(df)
        assert len(result) == 30

    def test_zero_values(self):
        """Zero values are handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": [0] * 30,
        })
        result = validate(df)
        assert len(result) == 30

    def test_mixed_positive_negative(self):
        """Mixed positive and negative values are handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": list(range(-15, 15)),
        })
        result = validate(df)
        assert len(result) == 30


class TestManySeries:
    """Test with many series."""

    def test_many_series(self):
        """Handle many series efficiently."""
        dfs = []
        for i in range(100):
            df = pd.DataFrame({
                "unique_id": [f"series_{i}"] * 30,
                "ds": pd.date_range("2024-01-01", periods=30),
                "y": np.random.randn(30).cumsum(),
            })
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        result = validate(df)
        assert len(result) == 3000


class TestNonNumericValues:
    """Test non-numeric values in target."""

    def test_string_in_target(self):
        """String in target column should be allowed through validate.

        Actual conversion errors will happen at model fitting time.
        """
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "y": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })
        # Validate should pass - type checking is at model level
        result = validate(df)
        assert len(result) == 10


class TestCustomColumnNames:
    """Test with custom column names."""

    def test_custom_columns_rejected(self):
        """Custom column names are rejected by fixed contract."""
        df = pd.DataFrame({
            "series_id": ["A"] * 30,
            "timestamp": pd.date_range("2024-01-01", periods=30),
            "value": range(30),
        })
        with pytest.raises(EContract, match="Missing required columns"):
            validate(df)


class TestFrequencyHandling:
    """Test different frequencies."""

    def test_hourly_frequency(self):
        """Hourly frequency is handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 48,
            "ds": pd.date_range("2024-01-01", periods=48, freq="h"),
            "y": range(48),
        })
        config = ForecastConfig(h=24, freq="h")
        result = validate(df, config)
        assert len(result) == 48

    def test_weekly_frequency(self):
        """Weekly frequency is handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 52,
            "ds": pd.date_range("2024-01-01", periods=52, freq="W"),
            "y": range(52),
        })
        config = ForecastConfig(h=12, freq="W")
        result = validate(df, config)
        assert len(result) == 52

    def test_monthly_frequency(self):
        """Monthly frequency is handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 24,
            "ds": pd.date_range("2024-01-01", periods=24, freq="ME"),
            "y": range(24),
        })
        config = ForecastConfig(h=6, freq="ME")
        result = validate(df, config)
        assert len(result) == 24


class TestConstantValues:
    """Test constant value series."""

    def test_constant_value(self):
        """Constant value series is handled."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30),
            "y": [5.0] * 30,
        })
        result = validate(df)
        assert len(result) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
