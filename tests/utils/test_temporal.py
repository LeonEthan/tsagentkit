"""Tests for tsagentkit.utils.temporal – drop_future_rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsagentkit.utils.temporal import drop_future_rows

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDropFutureRows:
    def test_drops_trailing_null_y(self) -> None:
        """Rows with null y beyond the last observed timestamp should be dropped."""
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
                "y": [1.0, 2.0, 3.0, np.nan, np.nan],
            }
        )
        cleaned, info = drop_future_rows(df)
        assert len(cleaned) == 3
        assert info is not None
        assert info["count"] == 2
        assert info["type"] == "future_rows_dropped"

    def test_preserves_internal_nan(self) -> None:
        """NaN in the middle of the series should NOT be dropped."""
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
                "y": [1.0, np.nan, 3.0, 4.0, 5.0],
            }
        )
        cleaned, info = drop_future_rows(df)
        assert len(cleaned) == 5
        assert info is None  # nothing dropped

    def test_no_future_rows(self) -> None:
        """All y values present → nothing dropped."""
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
                "y": [1.0, 2.0, 3.0],
            }
        )
        cleaned, info = drop_future_rows(df)
        assert len(cleaned) == 3
        assert info is None

    def test_multiple_series(self) -> None:
        """Each series should be handled independently."""
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A", "A", "B", "B", "B"],
                "ds": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ),
                "y": [1.0, 2.0, np.nan, 1.0, np.nan, np.nan],
            }
        )
        cleaned, info = drop_future_rows(df)
        # A: drops last row (1 future), B: drops last 2 rows (2 future)
        assert len(cleaned) == 3  # A has 2 rows, B has 1 row
        assert info is not None
        assert info["count"] == 3

    def test_missing_y_column(self) -> None:
        """If y column is missing, return a copy unchanged."""
        df = pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": pd.to_datetime(["2024-01-01"]),
                "value": [1.0],
            }
        )
        cleaned, info = drop_future_rows(df)
        assert len(cleaned) == 1
        assert info is None

    def test_missing_id_column(self) -> None:
        """If id column is missing, return a copy unchanged."""
        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01"]),
                "y": [1.0],
            }
        )
        cleaned, info = drop_future_rows(df)
        assert len(cleaned) == 1
        assert info is None

    def test_all_y_null_drops_everything(self) -> None:
        """Series with all-null y should have all rows dropped."""
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
                "y": [np.nan, np.nan],
            }
        )
        cleaned, info = drop_future_rows(df)
        assert len(cleaned) == 0
        assert info is not None
        assert info["count"] == 2
