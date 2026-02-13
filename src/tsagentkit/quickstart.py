"""Zero-config forecasting utilities for tsagentkit.

Provides ``forecast()`` and ``diagnose()`` convenience functions that
hide pipeline complexity behind a simple API.

Usage:
    >>> from tsagentkit.quickstart import forecast, diagnose
    >>> result = forecast(df, horizon=7)
    >>> report = diagnose(df)
"""

from __future__ import annotations

from typing import Literal, cast

import pandas as pd

from tsagentkit.contracts import DryRunResult, RunArtifact, TaskSpec
from tsagentkit.contracts.task_spec import COLUMN_ALIASES


def forecast(
    df: pd.DataFrame,
    horizon: int,
    freq: str | None = None,
    *,
    mode: Literal["quick", "standard", "strict"] = "standard",
) -> RunArtifact:
    """Zero-config forecasting: validate, plan, fit, predict in one call.

    Automatically renames columns to canonical names if common aliases
    are detected, sorts the data, infers frequency, and runs the full
    pipeline using ``TaskSpec.starter()``.

    Args:
        df: Input DataFrame. Must contain time-series panel data with
            columns mappable to ``unique_id``, ``ds``, ``y``.
        horizon: Number of future time steps to forecast.
        freq: Pandas frequency alias (e.g. ``"D"``, ``"H"``). If *None*,
            frequency is inferred from the data.
        mode: Execution mode — ``"quick"``, ``"standard"``, or ``"strict"``.

    Returns:
        A ``RunArtifact`` containing forecast results, provenance, etc.
    """
    from tsagentkit.serving.orchestration import run_forecast

    result = _prepare(df)
    spec = _build_task_spec(horizon, freq)

    return cast(RunArtifact, run_forecast(result, spec, mode=mode))


def diagnose(
    df: pd.DataFrame,
    *,
    freq: str | None = None,
    horizon: int = 1,
) -> dict[str, object]:
    """Run validation and QA on a DataFrame and return a structured report.

    This does **not** fit any models — it only checks data quality and
    produces a routing plan.

    Args:
        df: Input DataFrame (same format as ``forecast``).
        freq: Pandas frequency alias. Inferred if *None*.
        horizon: Forecast horizon used for QA checks (default 1).

    Returns:
        A dictionary with keys ``validation``, ``qa_report``, ``plan``,
        ``route_decision``, and ``task_spec_used``.
    """
    from tsagentkit.serving.orchestration import run_forecast

    result = _prepare(df)
    spec = _build_task_spec(horizon, freq)

    dry_result = run_forecast(result, spec, dry_run=True)
    if not isinstance(dry_result, DryRunResult):
        raise TypeError("Expected DryRunResult when run_forecast is called with dry_run=True.")
    return dry_result.to_dict()


def _build_task_spec(horizon: int, freq: str | None = None) -> TaskSpec:
    """Build TaskSpec with appropriate frequency settings.

    Args:
        horizon: Forecast horizon
        freq: Pandas frequency alias, or None for inference

    Returns:
        TaskSpec instance configured for the task
    """
    spec_kwargs: dict[str, object] = {"h": horizon}
    if freq is not None:
        spec_kwargs["freq"] = freq
    else:
        spec_kwargs["freq"] = "D"  # starter default

    spec = TaskSpec.starter(**spec_kwargs)

    # If freq was not provided, allow inference
    if freq is None:
        spec = spec.model_copy(update={"freq": None, "infer_freq": True})

    return spec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Copy, auto-rename columns, sort, and coerce ds to datetime."""
    result = df.copy()

    # Auto-rename common aliases
    rename_map: dict[str, str] = {}
    existing = set(result.columns)
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in existing:
            continue
        for alias in aliases:
            if alias in existing:
                rename_map[alias] = canonical
                existing.discard(alias)
                existing.add(canonical)
                break

    if rename_map:
        result = result.rename(columns=rename_map)

    # Coerce ds to datetime
    if "ds" in result.columns:
        result["ds"] = pd.to_datetime(result["ds"])

    # Sort
    if {"unique_id", "ds"}.issubset(result.columns):
        result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return result


__all__ = ["forecast", "diagnose"]
