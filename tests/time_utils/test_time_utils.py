"""Tests for tsagentkit.time – infer_freq, normalize_pandas_freq, make_regular_grid, make_future_index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit.contracts.errors import EFreqInferFail
from tsagentkit.time import (
    infer_freq,
    make_future_index,
    make_regular_grid,
    normalize_pandas_freq,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    freq: str = "D",
    n_series: int = 2,
    n_steps: int = 30,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a deterministic panel DataFrame with the given frequency."""
    rows: list[dict] = []
    for uid_idx in range(n_series):
        uid = f"series_{uid_idx}"
        dates = pd.date_range(start, periods=n_steps, freq=freq)
        for ds in dates:
            rows.append({"unique_id": uid, "ds": ds, "y": float(uid_idx + 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# infer_freq
# ---------------------------------------------------------------------------


class TestInferFreq:
    def test_daily(self) -> None:
        panel = _make_panel(freq="D")
        assert infer_freq(panel) == "D"

    def test_hourly(self) -> None:
        panel = _make_panel(freq="h", n_steps=48)
        freq = infer_freq(panel)
        assert freq in ("h", "H")

    def test_monthly(self) -> None:
        panel = _make_panel(freq="MS", n_steps=12)
        assert infer_freq(panel) == "MS"

    def test_ambiguous_raises(self) -> None:
        """Irregular timestamps should raise EFreqInferFail."""
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A", "A"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-08"]),
                "y": [1, 2, 3],
            }
        )
        with pytest.raises(EFreqInferFail):
            infer_freq(df)

    def test_single_row_series_raises(self) -> None:
        """Series with < 2 rows can't infer freq."""
        df = pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": pd.to_datetime(["2024-01-01"]),
                "y": [1.0],
            }
        )
        with pytest.raises(EFreqInferFail):
            infer_freq(df)


# ---------------------------------------------------------------------------
# normalize_pandas_freq
# ---------------------------------------------------------------------------


class TestNormalizePandasFreq:
    def test_m_to_me(self) -> None:
        assert normalize_pandas_freq("M") == "ME"

    def test_2m_to_2me(self) -> None:
        assert normalize_pandas_freq("2M") == "2ME"

    def test_daily_unchanged(self) -> None:
        assert normalize_pandas_freq("D") == "D"

    def test_hourly_unchanged(self) -> None:
        assert normalize_pandas_freq("h") == "h"

    def test_empty_string(self) -> None:
        assert normalize_pandas_freq("") == ""

    def test_me_unchanged(self) -> None:
        assert normalize_pandas_freq("ME") == "ME"


# ---------------------------------------------------------------------------
# make_regular_grid
# ---------------------------------------------------------------------------


class TestMakeRegularGrid:
    def test_fills_gaps_with_ffill(self) -> None:
        # Create panel with a gap
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": dates,
                "y": [1.0, 2.0, 4.0],
            }
        )
        result = make_regular_grid(df, freq="D", fill_policy="ffill")
        a_rows = result[result["unique_id"] == "A"]
        assert len(a_rows) == 4  # gap on Jan 3 filled
        assert a_rows["y"].notna().all()

    def test_fills_gaps_with_zero(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 2,
                "ds": dates,
                "y": [1.0, 3.0],
            }
        )
        result = make_regular_grid(df, freq="D", fill_policy="zero")
        a_rows = result[result["unique_id"] == "A"]
        assert len(a_rows) == 3
        jan2 = a_rows[a_rows["ds"] == pd.Timestamp("2024-01-02")]
        assert float(jan2["y"].iloc[0]) == 0.0

    def test_fills_gaps_with_mean(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 2,
                "ds": dates,
                "y": [2.0, 4.0],
            }
        )
        result = make_regular_grid(df, freq="D", fill_policy="mean")
        a_rows = result[result["unique_id"] == "A"]
        assert len(a_rows) == 3
        jan2 = a_rows[a_rows["ds"] == pd.Timestamp("2024-01-02")]
        assert float(jan2["y"].iloc[0]) == 3.0  # mean of 2 and 4

    def test_no_gap_returns_same_length(self) -> None:
        panel = _make_panel(freq="D", n_series=1, n_steps=5)
        result = make_regular_grid(panel, freq="D", fill_policy="ffill")
        assert len(result) == 5

    def test_multiple_series(self) -> None:
        panel = _make_panel(freq="D", n_series=3, n_steps=10)
        result = make_regular_grid(panel, freq="D", fill_policy="ffill")
        assert len(result) == 30


# ---------------------------------------------------------------------------
# make_future_index
# ---------------------------------------------------------------------------


class TestMakeFutureIndex:
    def test_generates_correct_future_dates(self) -> None:
        panel = _make_panel(freq="D", n_series=1, n_steps=10)
        future = make_future_index(panel, h=3, freq="D")
        assert len(future) == 3
        last_historical = panel["ds"].max()
        assert future["ds"].min() > last_historical

    def test_multiple_series(self) -> None:
        panel = _make_panel(freq="D", n_series=3, n_steps=5)
        future = make_future_index(panel, h=2, freq="D")
        assert len(future) == 6  # 3 series * 2 steps
        assert set(future["unique_id"]) == {"series_0", "series_1", "series_2"}

    def test_respects_horizon(self) -> None:
        panel = _make_panel(freq="D", n_series=1, n_steps=5)
        future = make_future_index(panel, h=7, freq="D")
        assert len(future) == 7

    def test_empty_panel(self) -> None:
        empty = pd.DataFrame(columns=["unique_id", "ds", "y"])
        future = make_future_index(empty, h=3, freq="D")
        assert future.empty
