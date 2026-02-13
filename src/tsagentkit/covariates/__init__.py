"""Covariate typing, alignment, and guardrails."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tsagentkit.contracts import (
    CovariateSpec,
    ECovariateIncompleteKnown,
    ECovariateLeakage,
    ECovariateStaticInvalid,
    ETaskSpecInvalid,
    TaskSpec,
)
from tsagentkit.time import make_future_index


@dataclass(frozen=True)
class CovariateBundle:
    static_x: pd.DataFrame | None = None
    past_x: pd.DataFrame | None = None
    future_x: pd.DataFrame | None = None


@dataclass(frozen=True)
class AlignedDataset:
    panel: pd.DataFrame
    static_x: pd.DataFrame | None
    past_x: pd.DataFrame | None
    future_x: pd.DataFrame | None
    covariate_spec: CovariateSpec | None
    future_index: pd.DataFrame | None


def align_covariates(
    panel: pd.DataFrame,
    task_spec: TaskSpec,
    covariates: CovariateBundle | None = None,
) -> AlignedDataset:
    """Align covariates and enforce coverage/leakage guardrails."""
    contract = task_spec.panel_contract
    uid_col = contract.unique_id_col
    ds_col = contract.ds_col
    y_col = contract.y_col

    panel = panel.copy()
    future_index = make_future_index(panel, task_spec.h, task_spec.freq, uid_col, ds_col, y_col)

    if covariates is not None:
        static_x = _validate_static_covariates(covariates.static_x, uid_col)
        past_x = _validate_past_covariates(covariates.past_x, future_index, uid_col, ds_col)
        future_x = _validate_future_covariates(
            covariates.future_x, future_index, uid_col, ds_col
        )
        return AlignedDataset(
            panel=_panel_base(panel, uid_col, ds_col, y_col),
            static_x=static_x,
            past_x=past_x,
            future_x=future_x,
            covariate_spec=task_spec.covariates,
            future_index=future_index,
        )

    covariate_cols = [c for c in panel.columns if c not in {uid_col, ds_col, y_col}]
    if not covariate_cols or task_spec.covariate_policy == "ignore":
        return AlignedDataset(
            panel=_panel_base(panel, uid_col, ds_col, y_col),
            static_x=None,
            past_x=None,
            future_x=None,
            covariate_spec=task_spec.covariates,
            future_index=future_index,
        )

    static_cols: list[str] = []
    past_cols: list[str] = []
    future_cols: list[str] = []

    policy = task_spec.covariate_policy
    spec = task_spec.covariates

    if policy == "spec" and spec is None and covariate_cols:
        raise ETaskSpecInvalid(
            "covariate_policy='spec' requires task_spec.covariates.",
            context={"missing": "covariates"},
        )

    if policy == "spec" and spec is not None:
        _validate_spec_roles(spec, covariate_cols)
        for col, role in spec.roles.items():
            if role == "static":
                static_cols.append(col)
            elif role == "past":
                past_cols.append(col)
            elif role == "future_known":
                future_cols.append(col)
    elif policy == "known":
        future_cols = covariate_cols
    elif policy == "observed":
        past_cols = covariate_cols
    elif policy == "auto":
        for col in covariate_cols:
            if _has_future_values(panel, future_index, col, uid_col, ds_col):
                # Candidate future-known: require full coverage
                future_cols.append(col)
                _enforce_future_coverage(panel, future_index, col, uid_col, ds_col)
            else:
                past_cols.append(col)
    else:
        past_cols = covariate_cols

    for col in future_cols:
        _enforce_future_coverage(panel, future_index, col, uid_col, ds_col)

    static_x = _extract_static(panel, uid_col, static_cols)
    past_x = _extract_time_covariates(panel, uid_col, ds_col, past_cols)
    future_x = _extract_future_covariates(panel, future_index, uid_col, ds_col, future_cols)

    _enforce_past_leakage(panel, future_index, uid_col, ds_col, past_cols)

    return AlignedDataset(
        panel=_panel_base(panel, uid_col, ds_col, y_col),
        static_x=static_x,
        past_x=past_x,
        future_x=future_x,
        covariate_spec=spec,
        future_index=future_index,
    )


def _panel_base(panel: pd.DataFrame, uid_col: str, ds_col: str, y_col: str) -> pd.DataFrame:
    cols = [c for c in [uid_col, ds_col, y_col] if c in panel.columns]
    return panel[cols].copy()


def _validate_spec_roles(spec: CovariateSpec, covariate_cols: list[str]) -> None:
    if not spec.roles:
        if covariate_cols:
            raise ETaskSpecInvalid(
                "covariate_policy='spec' requires explicit roles for all covariate columns.",
                context={"missing_roles_for": sorted(covariate_cols)},
            )
        return

    missing_in_panel = [col for col in spec.roles if col not in covariate_cols]
    if missing_in_panel:
        raise ETaskSpecInvalid(
            "CovariateSpec roles include columns not present in panel data.",
            context={"missing_in_panel": sorted(missing_in_panel)},
        )

    extra_in_panel = [col for col in covariate_cols if col not in spec.roles]
    if extra_in_panel:
        raise ETaskSpecInvalid(
            "covariate_policy='spec' requires roles for all panel covariates.",
            context={"missing_roles_for": sorted(extra_in_panel)},
        )


def _has_future_values(
    panel: pd.DataFrame,
    future_index: pd.DataFrame,
    col: str,
    uid_col: str,
    ds_col: str,
) -> bool:
    merged = future_index.merge(
        panel[[uid_col, ds_col, col]],
        on=[uid_col, ds_col],
        how="left",
    )
    return merged[col].notna().any()


def _merge_and_validate_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    merge_cols: list[str],
    covariate_cols: list[str],
    on_missing: type[Exception],
    error_msg_template: str,
    error_context_key: str,
) -> pd.DataFrame:
    """Merge two DataFrames and validate that covariate columns have no missing values.

    Args:
        left: Left DataFrame (typically future_index)
        right: Right DataFrame with covariate columns
        merge_cols: Columns to merge on
        covariate_cols: Covariate columns to validate for missing values
        on_missing: Exception class to raise if missing values found
        error_msg_template: Template string for error message (must have {col} placeholder)
        error_context_key: Key name for the context dictionary (e.g., "missing" or "future_values_count")

    Returns:
        Merged DataFrame

    Raises:
        Exception: Instance of `on_missing` if any covariate has missing values
    """
    merged = left.merge(right, on=merge_cols, how="left")
    for col in covariate_cols:
        if merged[col].isna().any():
            count = int(merged[col].isna().sum())
            raise on_missing(
                error_msg_template.format(col=col, count=count),
                context={"covariate": col, error_context_key: count},
            )
    return merged


def _enforce_future_coverage(
    panel: pd.DataFrame,
    future_index: pd.DataFrame,
    col: str,
    uid_col: str,
    ds_col: str,
) -> None:
    """Enforce that future-known covariate has full coverage in horizon."""
    _merge_and_validate_coverage(
        left=future_index,
        right=panel[[uid_col, ds_col, col]],
        merge_cols=[uid_col, ds_col],
        covariate_cols=[col],
        on_missing=ECovariateIncompleteKnown,
        error_msg_template="Future-known covariate '{col}' missing {count} values in horizon.",
        error_context_key="missing",
    )


def _enforce_past_leakage(
    panel: pd.DataFrame,
    future_index: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    past_cols: list[str],
) -> None:
    if not past_cols:
        return

    merged = future_index.merge(
        panel[[uid_col, ds_col] + past_cols],
        on=[uid_col, ds_col],
        how="left",
    )
    for col in past_cols:
        if merged[col].notna().any():
            count = int(merged[col].notna().sum())
            raise ECovariateLeakage(
                f"Past covariate '{col}' has {count} values in forecast horizon.",
                context={"covariate": col, "future_values_count": count},
            )


def _extract_static(panel: pd.DataFrame, uid_col: str, cols: list[str]) -> pd.DataFrame | None:
    if not cols:
        return None
    df = panel[[uid_col] + cols].dropna(subset=[uid_col]).copy()
    # Validate constant per unique_id
    for col in cols:
        counts = df.groupby(uid_col)[col].nunique(dropna=True)
        if (counts > 1).any():
            bad = counts[counts > 1].index.tolist()[:5]
            raise ECovariateStaticInvalid(
                f"Static covariate '{col}' varies within series.",
                context={"covariate": col, "unique_id_examples": bad},
            )
    return df.groupby(uid_col, as_index=False).first()


def _extract_time_covariates(
    panel: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    cols: list[str],
) -> pd.DataFrame | None:
    if not cols:
        return None
    return panel[[uid_col, ds_col] + cols].copy()


def _extract_future_covariates(
    panel: pd.DataFrame,
    future_index: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    cols: list[str],
) -> pd.DataFrame | None:
    if not cols:
        return None
    return _merge_and_validate_coverage(
        left=future_index,
        right=panel[[uid_col, ds_col] + cols],
        merge_cols=[uid_col, ds_col],
        covariate_cols=cols,
        on_missing=ECovariateIncompleteKnown,
        error_msg_template="Future-known covariate '{col}' missing {count} values in horizon.",
        error_context_key="missing",
    )


def _validate_static_covariates(
    static_x: pd.DataFrame | None,
    uid_col: str,
) -> pd.DataFrame | None:
    if static_x is None or static_x.empty:
        return None
    counts = static_x.groupby(uid_col).size()
    if (counts != 1).any():
        bad = counts[counts != 1].index.tolist()[:5]
        raise ECovariateStaticInvalid(
            "Static covariates must have exactly one row per unique_id.",
            context={"unique_id_examples": bad},
        )
    return static_x.copy()


def _validate_past_covariates(
    past_x: pd.DataFrame | None,
    future_index: pd.DataFrame,
    uid_col: str,
    ds_col: str,
) -> pd.DataFrame | None:
    if past_x is None or past_x.empty:
        return None
    merged = future_index.merge(
        past_x,
        on=[uid_col, ds_col],
        how="left",
    )
    covariate_cols = [c for c in past_x.columns if c not in {uid_col, ds_col}]
    for col in covariate_cols:
        if merged[col].notna().any():
            count = int(merged[col].notna().sum())
            raise ECovariateLeakage(
                f"Past covariate '{col}' has {count} values in forecast horizon.",
                context={"covariate": col, "future_values_count": count},
            )
    return past_x.copy()


def _validate_future_covariates(
    future_x: pd.DataFrame | None,
    future_index: pd.DataFrame,
    uid_col: str,
    ds_col: str,
) -> pd.DataFrame | None:
    if future_x is None or future_x.empty:
        return None
    covariate_cols = [c for c in future_x.columns if c not in {uid_col, ds_col}]
    _merge_and_validate_coverage(
        left=future_index,
        right=future_x,
        merge_cols=[uid_col, ds_col],
        covariate_cols=covariate_cols,
        on_missing=ECovariateIncompleteKnown,
        error_msg_template="Future-known covariate '{col}' missing {count} values in horizon.",
        error_context_key="missing",
    )
    return future_x.copy()


__all__ = [
    "AlignedDataset",
    "CovariateBundle",
    "align_covariates",
]
