"""Temporal utilities for cleaning time series data."""

from __future__ import annotations

import pandas as pd


def drop_future_rows(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
) -> tuple[pd.DataFrame, dict[str, int | str] | None]:
    """Drop rows where target is missing beyond the last observed timestamp.

    Keeps missing values inside the observed history for QA/repair.
    """
    if y_col not in df.columns or ds_col not in df.columns or id_col not in df.columns:
        return df.copy(), None

    data = df.copy()

    # Compute last observed timestamp per series.
    last_observed = data[data[y_col].notna()].groupby(id_col)[ds_col].max()
    data["_last_observed"] = data[id_col].map(last_observed)

    future_mask = data[y_col].isna() & (
        data["_last_observed"].isna() | (data[ds_col] > data["_last_observed"])
    )
    dropped = int(future_mask.sum())

    cleaned = data.loc[~future_mask].drop(columns=["_last_observed"])

    if dropped == 0:
        return cleaned, None

    return cleaned, {
        "type": "future_rows_dropped",
        "count": dropped,
        "rule": "y_null_future",
    }
