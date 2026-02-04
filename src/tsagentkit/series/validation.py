"""Data validation schemas.

Provides validation functions to check input data against the required
schema for time series forecasting.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tsagentkit.contracts.errors import (
    EContractDuplicateKey,
    EContractInvalid,
    EContractInvalidType,
    EContractMissingColumn,
    EContractUnsorted,
    EFreqInferFail,
)
from tsagentkit.contracts.results import ValidationReport
from tsagentkit.contracts.task_spec import PanelContract


def normalize_panel_columns(
    df: pd.DataFrame,
    contract: PanelContract,
) -> tuple[pd.DataFrame, dict[str, str] | None]:
    """Normalize panel columns to the canonical contract names.

    Returns:
        (normalized_df, column_map) where column_map maps original names to
        canonical names. If no normalization is needed, column_map is None.
    """
    default_contract = PanelContract()
    mapping = {
        contract.unique_id_col: default_contract.unique_id_col,
        contract.ds_col: default_contract.ds_col,
        contract.y_col: default_contract.y_col,
    }

    if mapping == {
        default_contract.unique_id_col: default_contract.unique_id_col,
        default_contract.ds_col: default_contract.ds_col,
        default_contract.y_col: default_contract.y_col,
    }:
        return df, None

    missing = [src for src in mapping if src not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.rename(columns=mapping), mapping


def validate_contract(
    data: Any,
    panel_contract: PanelContract | None = None,
    apply_aggregation: bool = False,
    return_data: bool = False,
) -> ValidationReport | tuple[ValidationReport, pd.DataFrame]:
    """Validate input data against the required schema.

    Args:
        data: Input data (DataFrame or convertible)
        panel_contract: Optional PanelContract specifying column names and aggregation
        apply_aggregation: Whether to aggregate duplicates when allowed by contract
        return_data: If True, return (ValidationReport, normalized_df)

    Returns:
        ValidationReport (and optionally normalized DataFrame)
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    contract = panel_contract or PanelContract()
    uid_col = contract.unique_id_col
    ds_col = contract.ds_col
    y_col = contract.y_col

    # Convert to DataFrame if needed
    df = _convert_to_dataframe(data)
    if df is None:
        errors.append({
            "code": EContractInvalid.error_code,
            "message": "Data must be a DataFrame or convertible to DataFrame",
            "context": {"type": type(data).__name__},
        })
        report = ValidationReport(valid=False, errors=errors, warnings=warnings)
        return (report, pd.DataFrame()) if return_data else report

    # Check required columns
    required_cols = {uid_col, ds_col, y_col}
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
        report = ValidationReport(
            valid=False,
            errors=errors,
            warnings=warnings,
            stats={"num_rows": len(df)},
        )
        return (report, df) if return_data else report

    # Normalize types
    type_errors = _validate_column_types(df, uid_col, ds_col, y_col)
    errors.extend(type_errors)

    # Aggregate duplicates if allowed
    if contract.aggregation != "reject" and apply_aggregation:
        df, agg_warnings = _aggregate_duplicates(df, uid_col, ds_col, y_col, contract.aggregation)
        warnings.extend(agg_warnings)

    # Check for duplicates (post-aggregation if any)
    duplicate_errors = _validate_no_duplicates(df, uid_col, ds_col, contract.aggregation)
    errors.extend(duplicate_errors)

    # Check sorting
    sort_errors = _validate_sorted(df, uid_col, ds_col)
    errors.extend(sort_errors)

    # Try to infer frequency
    freq_warnings, freq_stats = _validate_frequency(df, uid_col, ds_col)
    warnings.extend(freq_warnings)
    stats.update(freq_stats)

    # Collect statistics (only if y is numeric)
    if not any(
        e["code"] == EContractInvalidType.error_code
        and e.get("context", {}).get("column") == y_col
        for e in errors
    ):
        stats.update(_compute_stats(df, uid_col, ds_col, y_col))

    valid = len(errors) == 0
    report = ValidationReport(valid=valid, errors=errors, warnings=warnings, stats=stats)

    return (report, df) if return_data else report


def _convert_to_dataframe(data: Any) -> pd.DataFrame | None:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        if hasattr(data, "to_pandas"):
            return data.to_pandas()
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data)
    except Exception:
        pass

    return None


def _validate_column_types(
    df: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    y_col: str,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    if not pd.api.types.is_string_dtype(df[uid_col]):
        try:
            df[uid_col] = df[uid_col].astype(str)
        except Exception as e:
            errors.append({
                "code": EContractInvalidType.error_code,
                "message": f"Column '{uid_col}' must be convertible to string",
                "context": {
                    "column": uid_col,
                    "actual_type": str(df[uid_col].dtype),
                    "error": str(e),
                },
            })

    if not pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        try:
            df[ds_col] = pd.to_datetime(df[ds_col], format="mixed")
        except Exception as e:
            errors.append({
                "code": EContractInvalidType.error_code,
                "message": f"Column '{ds_col}' must be convertible to datetime",
                "context": {
                    "column": ds_col,
                    "actual_type": str(df[ds_col].dtype),
                    "error": str(e),
                },
            })

    if len(df) > 0 and not pd.api.types.is_numeric_dtype(df[y_col]):
        errors.append({
            "code": EContractInvalidType.error_code,
            "message": f"Column '{y_col}' must be numeric",
            "context": {
                "column": y_col,
                "actual_type": str(df[y_col].dtype),
            },
        })

    return errors


def _aggregate_duplicates(
    df: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    y_col: str,
    aggregation: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []

    duplicates = df.duplicated(subset=[uid_col, ds_col], keep=False)
    if not duplicates.any():
        return df, warnings

    if aggregation == "reject":
        return df, warnings

    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "last": "last",
    }
    if aggregation not in agg_map:
        return df, warnings

    grouped = df.groupby([uid_col, ds_col], as_index=False)
    df = grouped.agg({y_col: agg_map[aggregation]})

    warnings.append({
        "code": "W_CONTRACT_AGGREGATED",
        "message": f"Aggregated duplicate keys using '{aggregation}'",
        "context": {"aggregation": aggregation},
    })

    return df, warnings


def _validate_no_duplicates(
    df: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    aggregation: str,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    duplicates = df.duplicated(subset=[uid_col, ds_col], keep=False)
    if duplicates.any():
        dup_df = df[duplicates]
        dup_keys = dup_df[[uid_col, ds_col]].drop_duplicates()

        errors.append({
            "code": EContractDuplicateKey.error_code,
            "message": f"Found {len(dup_keys)} duplicate ({uid_col}, {ds_col}) pairs",
            "context": {
                "num_duplicates": int(duplicates.sum()),
                "duplicate_keys": dup_keys.head(10).to_dict("records"),
                "aggregation": aggregation,
            },
        })

    return errors


def _validate_sorted(
    df: pd.DataFrame,
    uid_col: str,
    ds_col: str,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    expected_order = df.sort_values([uid_col, ds_col]).index
    if not df.index.equals(expected_order):
        errors.append({
            "code": EContractUnsorted.error_code,
            "message": f"Data must be sorted by ({uid_col}, {ds_col})",
            "context": {
                "suggestion": f"Use df.sort_values(['{uid_col}', '{ds_col}']) to fix",
            },
        })

    return errors


def _validate_frequency(
    df: pd.DataFrame,
    uid_col: str,
    ds_col: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    freq_counts: dict[str, int] = {}

    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid].sort_values(ds_col)
        if len(series) < 2:
            continue

        try:
            freq = pd.infer_freq(series[ds_col])
            if freq:
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
        except Exception:
            pass

    if freq_counts:
        inferred_freq = max(freq_counts, key=freq_counts.get)
        stats["inferred_freq"] = inferred_freq
        stats["freq_counts"] = freq_counts
    else:
        warnings.append({
            "code": EFreqInferFail.error_code,
            "message": "Could not infer frequency from data",
            "context": {
                "suggestion": "Specify frequency explicitly in TaskSpec",
            },
        })

    return warnings, stats


def _compute_stats(
    df: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    y_col: str,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "num_rows": len(df),
        "num_series": df[uid_col].nunique(),
    }

    if pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        stats["date_range"] = {
            "start": df[ds_col].min().isoformat() if not df[ds_col].empty else None,
            "end": df[ds_col].max().isoformat() if not df[ds_col].empty else None,
        }
    else:
        stats["date_range"] = {"start": None, "end": None}

    if len(df) > 0 and pd.api.types.is_numeric_dtype(df[y_col]):
        stats["y_stats"] = {
            "mean": float(df[y_col].mean()),
            "std": float(df[y_col].std()),
            "min": float(df[y_col].min()),
            "max": float(df[y_col].max()),
            "missing": int(df[y_col].isna().sum()),
        }
    else:
        stats["y_stats"] = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "missing": int(df[y_col].isna().sum()) if y_col in df.columns else 0,
        }

    if len(df) > 0:
        series_lengths = df.groupby(uid_col).size()
        stats["series_lengths"] = {
            "min": int(series_lengths.min()),
            "max": int(series_lengths.max()),
            "mean": float(series_lengths.mean()),
        }
    else:
        stats["series_lengths"] = {"min": 0, "max": 0, "mean": 0.0}

    return stats


__all__ = ["validate_contract", "normalize_panel_columns"]
