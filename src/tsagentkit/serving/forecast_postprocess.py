"""Shared forecast post-processing helpers for serving paths."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tsagentkit.utils import normalize_quantile_columns


def resolve_model_name(artifact: Any) -> str | None:
    """Resolve model name from a model artifact-like object."""
    if artifact is None:
        return None
    model_name = getattr(artifact, "model_name", None)
    if model_name is None and hasattr(artifact, "metadata"):
        metadata = getattr(artifact, "metadata", None) or {}
        if isinstance(metadata, dict):
            model_name = metadata.get("model_name")
    return model_name


def add_model_column(
    forecast: pd.DataFrame,
    model_name: str | None,
) -> pd.DataFrame:
    """Ensure forecast includes a model column."""
    if "model" in forecast.columns:
        return forecast
    updated = forecast.copy()
    updated["model"] = model_name or "model"
    return updated


def maybe_reconcile_forecast(
    forecast: pd.DataFrame,
    *,
    dataset: Any,
    plan: Any | None,
    reconciliation_method: str,
) -> pd.DataFrame:
    """Apply hierarchical reconciliation when plan and hierarchy are present."""
    if plan is None:
        return forecast
    if not (dataset.is_hierarchical() and dataset.hierarchy):
        return forecast

    from tsagentkit.hierarchy import apply_reconciliation_if_needed

    return apply_reconciliation_if_needed(
        forecast=forecast,
        hierarchy=dataset.hierarchy,
        method=reconciliation_method,
    )


def normalize_and_sort_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    """Normalize quantile columns and sort by canonical keys."""
    normalized = normalize_quantile_columns(forecast)
    if {"unique_id", "ds"}.issubset(normalized.columns):
        normalized = normalized.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return normalized

