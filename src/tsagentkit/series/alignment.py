"""Time alignment and resampling utilities.

Provides timezone unification and resampling for time series data.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def align_timezone(
    df: pd.DataFrame,
    target_tz: str | None = "UTC",
    ds_col: str = "ds",
) -> pd.DataFrame:
    """Unify timezones across a dataset.

    Converts all datetime values to the target timezone. Handles
    timezone-aware and timezone-naive datetimes appropriately.

    Args:
        df: DataFrame with datetime column
        target_tz: Target timezone (default: "UTC", None for naive)
        ds_col: Name of datetime column (default: "ds")

    Returns:
        DataFrame with unified timezone

    Raises:
        ValueError: If ds_col is not found or not datetime
    """
    if ds_col not in df.columns:
        raise ValueError(f"Column '{ds_col}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        raise ValueError(f"Column '{ds_col}' must be datetime type")

    result = df.copy()

    # Handle timezone
    if target_tz is None:
        # Make timezone-naive
        if result[ds_col].dt.tz is not None:
            result[ds_col] = result[ds_col].dt.tz_localize(None)
    else:
        # Convert to target timezone
        if result[ds_col].dt.tz is None:
            # Assume UTC for naive datetimes, then convert
            result[ds_col] = result[ds_col].dt.tz_localize("UTC").dt.tz_convert(target_tz)
        else:
            result[ds_col] = result[ds_col].dt.tz_convert(target_tz)

    return result


def resample_series(
    df: pd.DataFrame,
    freq: str,
    agg_func: Literal["sum", "mean", "last", "first", "max", "min"] = "sum",
    ds_col: str = "ds",
    unique_id_col: str = "unique_id",
    y_col: str = "y",
) -> pd.DataFrame:
    """Resample time series to a new frequency.

    Resamples each series independently to the target frequency using
    the specified aggregation function.

    Args:
        df: DataFrame with time series data
        freq: Target frequency (pandas freq string, e.g., 'D', 'H', 'M')
        agg_func: Aggregation function (default: "sum")
        ds_col: Name of datetime column (default: "ds")
        unique_id_col: Name of series ID column (default: "unique_id")
        y_col: Name of target column (default: "y")

    Returns:
        Resampled DataFrame

    Raises:
        ValueError: If required columns not found
    """
    required_cols = {ds_col, unique_id_col, y_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        raise ValueError(f"Column '{ds_col}' must be datetime type")

    # Resample each series
    resampled_frames: list[pd.DataFrame] = []

    for uid in df[unique_id_col].unique():
        series = df[df[unique_id_col] == uid].set_index(ds_col).sort_index()

        # Select numeric columns for resampling
        numeric_cols = series.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            continue

        # Resample
        resampler = series[numeric_cols].resample(freq)

        # Apply aggregation
        if agg_func == "sum":
            resampled = resampler.sum()
        elif agg_func == "mean":
            resampled = resampler.mean()
        elif agg_func == "last":
            resampled = resampler.last()
        elif agg_func == "first":
            resampled = resampler.first()
        elif agg_func == "max":
            resampled = resampler.max()
        elif agg_func == "min":
            resampled = resampler.min()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

        # Add back unique_id
        resampled[unique_id_col] = uid
        resampled = resampled.reset_index()

        resampled_frames.append(resampled)

    if not resampled_frames:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=[unique_id_col, ds_col, y_col])

    # Combine all series
    result = pd.concat(resampled_frames, ignore_index=True)

    # Reorder columns to standard order
    cols = [unique_id_col, ds_col] + [c for c in result.columns if c not in {unique_id_col, ds_col}]

    return result[cols]


def fill_gaps(
    df: pd.DataFrame,
    freq: str,
    method: Literal["interpolate", "forward", "backward", "zero"] = "interpolate",
    ds_col: str = "ds",
    unique_id_col: str = "unique_id",
) -> pd.DataFrame:
    """Fill gaps in time series data.

    Identifies missing timestamps and fills them using the specified method.

    Args:
        df: DataFrame with time series data
        freq: Expected frequency (pandas freq string)
        method: Fill method (default: "interpolate")
        ds_col: Name of datetime column (default: "ds")
        unique_id_col: Name of series ID column (default: "unique_id")

    Returns:
        DataFrame with gaps filled
    """
    filled_frames: list[pd.DataFrame] = []

    for uid in df[unique_id_col].unique():
        series = df[df[unique_id_col] == uid].set_index(ds_col).sort_index()

        # Create complete date range
        full_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)

        # Reindex to include gaps
        series_filled = series.reindex(full_range)

        # Select only numeric columns for filling
        numeric_cols = series_filled.select_dtypes(include=["number"]).columns.tolist()

        # Fill missing values in numeric columns only
        if method == "interpolate":
            series_filled[numeric_cols] = series_filled[numeric_cols].interpolate(method="linear")
        elif method == "forward":
            series_filled[numeric_cols] = series_filled[numeric_cols].ffill()
        elif method == "backward":
            series_filled[numeric_cols] = series_filled[numeric_cols].bfill()
        elif method == "zero":
            series_filled[numeric_cols] = series_filled[numeric_cols].fillna(0)

        # Add back unique_id
        series_filled[unique_id_col] = uid
        series_filled = series_filled.reset_index()
        # Rename the datetime column (could be "index" or the original index name)
        if "index" in series_filled.columns:
            series_filled = series_filled.rename(columns={"index": ds_col})

        filled_frames.append(series_filled)

    if not filled_frames:
        return df.copy()

    result = pd.concat(filled_frames, ignore_index=True)

    # Reorder columns
    cols = [unique_id_col, ds_col] + [c for c in result.columns if c not in {unique_id_col, ds_col}]

    return result[cols]
