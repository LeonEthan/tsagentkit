"""Time utilities for frequency handling and index generation."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from tsagentkit.contracts.errors import EFreqInferFail


def infer_freq(
    panel: pd.DataFrame,
    id_col: str = "unique_id",
    ds_col: str = "ds",
) -> str:
    """Infer frequency from panel data."""
    freq_counts: dict[str, int] = {}

    for uid in panel[id_col].unique():
        series = panel[panel[id_col] == uid].sort_values(ds_col)
        if len(series) < 2:
            continue
        try:
            freq = pd.infer_freq(series[ds_col])
            if freq:
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
        except Exception:
            continue

    if not freq_counts:
        raise EFreqInferFail(
            "Frequency could not be inferred from panel data.",
            context={"unique_id_count": int(panel[id_col].nunique())},
        )

    return max(freq_counts, key=lambda k: freq_counts[k])


def normalize_pandas_freq(freq: str) -> str:
    """Normalize pandas frequency aliases to avoid deprecation warnings.

    Note: ME (month end) alias is not supported in all pandas versions,
    so we keep M which works in both pandas 1.x and 2.x.
    """
    # M and other standard aliases work in all pandas versions
    return freq


def make_regular_grid(
    panel: pd.DataFrame,
    freq: str,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    fill_policy: Literal["error", "ffill", "bfill", "zero", "mean"] = "error",
) -> pd.DataFrame:
    """Expand to a regular grid per-series with optional fill policy."""
    frames: list[pd.DataFrame] = []
    freq = normalize_pandas_freq(freq)

    for uid in panel[id_col].unique():
        series = panel[panel[id_col] == uid].copy()
        series = series.sort_values(ds_col)
        series = series.set_index(ds_col)
        full_range = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=freq,
        )
        series = series.reindex(full_range)
        series[id_col] = uid
        series = series.reset_index().rename(columns={"index": ds_col})

        if fill_policy != "error":
            numeric_cols = series.select_dtypes(include=["number"]).columns.tolist()
            if fill_policy == "ffill":
                series[numeric_cols] = series[numeric_cols].ffill()
            elif fill_policy == "bfill":
                series[numeric_cols] = series[numeric_cols].bfill()
            elif fill_policy == "zero":
                series[numeric_cols] = series[numeric_cols].fillna(0)
            elif fill_policy == "mean":
                means = series[numeric_cols].mean()
                series[numeric_cols] = series[numeric_cols].fillna(means)

        frames.append(series)

    if not frames:
        return panel.copy()

    return pd.concat(frames, ignore_index=True)


def make_future_index(
    panel: pd.DataFrame,
    h: int,
    freq: str,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
) -> pd.DataFrame:
    """Generate future index per series."""
    rows: list[pd.DataFrame] = []
    freq = normalize_pandas_freq(freq)

    for uid in panel[id_col].unique():
        series = panel[panel[id_col] == uid].copy()
        series = series.sort_values(ds_col)
        if y_col in series.columns and series[y_col].notna().any():
            last_observed = series.loc[series[y_col].notna(), ds_col].max()
        else:
            last_observed = series[ds_col].max()

        start = pd.Timestamp(last_observed) + pd.tseries.frequencies.to_offset(freq)
        future_ds = pd.date_range(start=start, periods=h, freq=freq)
        rows.append(pd.DataFrame({id_col: uid, ds_col: future_ds}))

    if not rows:
        return pd.DataFrame(columns=[id_col, ds_col])

    return pd.concat(rows, ignore_index=True)


__all__ = [
    "infer_freq",
    "normalize_pandas_freq",
    "make_regular_grid",
    "make_future_index",
]
