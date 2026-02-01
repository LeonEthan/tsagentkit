"""Utilities for quantile column normalization."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

_QUANTILE_PATTERN = re.compile(r"^q[_]?([0-9]+(?:\.[0-9]+)?)$")


def _format_quantile(q: float) -> str:
    text = f"{q:.6g}"
    if "e" in text or "E" in text:
        text = f"{q:.8f}".rstrip("0").rstrip(".")
    return text


def quantile_col_name(q: float) -> str:
    """Return canonical quantile column name (e.g., q0.1)."""
    return f"q{_format_quantile(q)}"


def parse_quantile_column(col: str) -> float | None:
    """Parse quantile value from column name."""
    match = _QUANTILE_PATTERN.match(col)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1:
        value = value / 100.0
    if not 0 < value < 1:
        return None
    return value


def extract_quantiles(columns: Iterable[str]) -> list[float]:
    """Extract sorted quantile values from column names."""
    values = []
    for col in columns:
        q = parse_quantile_column(col)
        if q is not None:
            values.append(q)
    return sorted(set(values))


def normalize_quantile_columns(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """Normalize quantile columns to canonical names (q0.1, q0.9, ...)."""
    if not inplace:
        df = df.copy()

    matched_cols: list[str] = []
    mapping: dict[float, list[str]] = {}

    for col in df.columns:
        q = parse_quantile_column(col)
        if q is None:
            continue
        matched_cols.append(col)
        mapping.setdefault(q, []).append(col)

    if not mapping:
        return df

    for q, cols in mapping.items():
        canonical = quantile_col_name(q)
        if canonical in df.columns and canonical not in cols:
            cols = [canonical] + cols
        if len(cols) == 1:
            df[canonical] = df[cols[0]]
        else:
            df[canonical] = df[cols].bfill(axis=1).iloc[:, 0]

    keep = {quantile_col_name(q) for q in mapping}
    drop_cols = [c for c in matched_cols if c not in keep]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df
