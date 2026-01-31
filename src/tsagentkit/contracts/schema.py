"""Data validation schemas.

Provides validation functions to check input data against the required
schema for time series forecasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .errors import (
    EContractDuplicateKey,
    EContractInvalidFrequency,
    EContractInvalidType,
    EContractMissingColumn,
    EContractUnsorted,
)
from .results import ValidationReport

if TYPE_CHECKING:
    pass


def validate_contract(data: Any) -> ValidationReport:
    """Validate input data against the required schema.

    Checks that the data:
    1. Is a DataFrame or convertible to one
    2. Has required columns: unique_id, ds, y
    3. Has correct types for each column
    4. Has no duplicate (unique_id, ds) pairs
    5. Is sorted by (unique_id, ds)
    6. Has a valid/inferrable frequency

    Args:
        data: Input data (DataFrame or convertible)

    Returns:
        ValidationReport with results and any errors/warnings

    Raises:
        EContractMissingColumn: If required columns are missing
        EContractInvalidType: If columns have wrong types
        EContractDuplicateKey: If duplicate keys exist
        EContractUnsorted: If data is not properly sorted
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    # Convert to DataFrame if needed
    df = _convert_to_dataframe(data)
    if df is None:
        errors.append({
            "code": "E_CONTRACT_INVALID_INPUT",
            "message": "Data must be a DataFrame or convertible to DataFrame",
            "context": {"type": type(data).__name__},
        })
        return ValidationReport(valid=False, errors=errors, warnings=warnings)

    # Check required columns
    required_cols = {"unique_id", "ds", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append({
            "code": EContractMissingColumn.error_code,
            "message": f"Missing required columns: {sorted(missing_cols)}",
            "context": {
                "missing": sorted(missing_cols),
                "available": sorted(df.columns.tolist()),
            },
        })
        # Can't continue without required columns
        return ValidationReport(
            valid=False, errors=errors, warnings=warnings, stats={"num_rows": len(df)}
        )

    # Check column types
    type_errors = _validate_column_types(df)
    errors.extend(type_errors)

    # Check for duplicates
    duplicate_errors = _validate_no_duplicates(df)
    errors.extend(duplicate_errors)

    # Check sorting
    sort_errors = _validate_sorted(df)
    errors.extend(sort_errors)

    # Try to infer frequency
    freq_warnings, freq_stats = _validate_frequency(df)
    warnings.extend(freq_warnings)
    stats.update(freq_stats)

    # Collect statistics (only if y is numeric)
    if not any(e["code"] == EContractInvalidType.error_code and
               e.get("context", {}).get("column") == "y"
               for e in errors):
        stats.update(_compute_stats(df))

    valid = len(errors) == 0
    return ValidationReport(
        valid=valid, errors=errors, warnings=warnings, stats=stats
    )


def _convert_to_dataframe(data: Any) -> pd.DataFrame | None:
    """Convert input data to DataFrame.

    Args:
        data: Input data of various types

    Returns:
        DataFrame or None if conversion fails
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    # Try common conversions
    try:
        if hasattr(data, "to_pandas"):  # Polars, Arrow, etc.
            return data.to_pandas()
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data)
    except Exception:
        pass

    return None


def _validate_column_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate that columns have correct types.

    Args:
        df: Input DataFrame

    Returns:
        List of error dictionaries
    """
    errors: list[dict[str, Any]] = []

    # unique_id should be string or convertible to string
    if not pd.api.types.is_string_dtype(df["unique_id"]):
        # Try to convert
        try:
            df["unique_id"] = df["unique_id"].astype(str)
        except Exception as e:
            errors.append({
                "code": EContractInvalidType.error_code,
                "message": "Column 'unique_id' must be convertible to string",
                "context": {
                    "column": "unique_id",
                    "actual_type": str(df["unique_id"].dtype),
                    "error": str(e),
                },
            })

    # ds should be datetime
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        try:
            df["ds"] = pd.to_datetime(df["ds"], format="mixed")
        except Exception as e:
            errors.append({
                "code": EContractInvalidType.error_code,
                "message": "Column 'ds' must be convertible to datetime",
                "context": {
                    "column": "ds",
                    "actual_type": str(df["ds"].dtype),
                    "error": str(e),
                },
            })

    # y should be numeric (skip if empty)
    if len(df) > 0 and not pd.api.types.is_numeric_dtype(df["y"]):
        errors.append({
            "code": EContractInvalidType.error_code,
            "message": "Column 'y' must be numeric",
            "context": {
                "column": "y",
                "actual_type": str(df["y"].dtype),
            },
        })

    return errors


def _validate_no_duplicates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check for duplicate (unique_id, ds) pairs.

    Args:
        df: Input DataFrame

    Returns:
        List of error dictionaries
    """
    errors: list[dict[str, Any]] = []

    # Check for duplicates
    duplicates = df.duplicated(subset=["unique_id", "ds"], keep=False)
    if duplicates.any():
        dup_df = df[duplicates]
        dup_keys = dup_df[["unique_id", "ds"]].drop_duplicates()

        errors.append({
            "code": EContractDuplicateKey.error_code,
            "message": f"Found {len(dup_keys)} duplicate (unique_id, ds) pairs",
            "context": {
                "num_duplicates": int(duplicates.sum()),
                "duplicate_keys": dup_keys.head(10).to_dict("records"),
            },
        })

    return errors


def _validate_sorted(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check if data is sorted by (unique_id, ds).

    Args:
        df: Input DataFrame

    Returns:
        List of error dictionaries
    """
    errors: list[dict[str, Any]] = []

    # Check if sorted
    expected_order = df.sort_values(["unique_id", "ds"]).index
    if not df.index.equals(expected_order):
        errors.append({
            "code": EContractUnsorted.error_code,
            "message": "Data must be sorted by (unique_id, ds)",
            "context": {
                "suggestion": "Use df.sort_values(['unique_id', 'ds']) to fix",
            },
        })

    return errors


def _validate_frequency(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Try to infer and validate frequency.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (warnings list, stats dict)
    """
    warnings: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    # Try to infer frequency from each series
    freq_counts: dict[str, int] = {}

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid].sort_values("ds")
        if len(series) < 2:
            continue

        try:
            freq = pd.infer_freq(series["ds"])
            if freq:
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
        except Exception:
            pass

    if freq_counts:
        # Take most common frequency
        inferred_freq = max(freq_counts, key=freq_counts.get)
        stats["inferred_freq"] = inferred_freq
        stats["freq_counts"] = freq_counts
    else:
        warnings.append({
            "code": EContractInvalidFrequency.error_code,
            "message": "Could not infer frequency from data",
            "context": {
                "suggestion": "Specify frequency explicitly in TaskSpec",
            },
        })

    return warnings, stats


def _compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute basic statistics about the data.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of statistics
    """
    stats: dict[str, Any] = {
        "num_rows": len(df),
        "num_series": df["unique_id"].nunique(),
    }

    # Date range (only if ds is datetime)
    if pd.api.types.is_datetime64_any_dtype(df["ds"]):
        stats["date_range"] = {
            "start": df["ds"].min().isoformat() if not df["ds"].empty else None,
            "end": df["ds"].max().isoformat() if not df["ds"].empty else None,
        }
    else:
        stats["date_range"] = {"start": None, "end": None}

    # Y stats (only if numeric and not empty)
    if len(df) > 0 and pd.api.types.is_numeric_dtype(df["y"]):
        stats["y_stats"] = {
            "mean": float(df["y"].mean()),
            "std": float(df["y"].std()),
            "min": float(df["y"].min()),
            "max": float(df["y"].max()),
            "missing": int(df["y"].isna().sum()),
        }
    else:
        stats["y_stats"] = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "missing": int(df["y"].isna().sum()) if "y" in df.columns else 0,
        }

    # Series lengths (only if we have data)
    if len(df) > 0:
        series_lengths = df.groupby("unique_id").size()
        stats["series_lengths"] = {
            "min": int(series_lengths.min()),
            "max": int(series_lengths.max()),
            "mean": float(series_lengths.mean()),
        }
    else:
        stats["series_lengths"] = {"min": 0, "max": 0, "mean": 0.0}

    return stats
