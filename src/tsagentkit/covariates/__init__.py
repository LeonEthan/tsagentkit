"""Covariate typing, alignment, and guardrails."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tsagentkit.contracts import (
    CovariateSpec,
    ECovariateIncompleteKnown,
    ECovariateLeakage,
    ECovariateStaticInvalid,
    TaskSpec,
    ETaskSpecInvalid,
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


def _enforce_future_coverage(
    panel: pd.DataFrame,
    future_index: pd.DataFrame,
    col: str,
    uid_col: str,
    ds_col: str,
) -> None:
    merged = future_index.merge(
        panel[[uid_col, ds_col, col]],
        on=[uid_col, ds_col],
        how="left",
    )
    if merged[col].isna().any():
        missing = int(merged[col].isna().sum())
        raise ECovariateIncompleteKnown(
            f"Future-known covariate '{col}' missing {missing} values in horizon.",
            context={"covariate": col, "missing": missing},
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
    merged = future_index.merge(
        panel[[uid_col, ds_col] + cols],
        on=[uid_col, ds_col],
        how="left",
    )
    for col in cols:
        if merged[col].isna().any():
            missing = int(merged[col].isna().sum())
            raise ECovariateIncompleteKnown(
                f"Future-known covariate '{col}' missing {missing} values in horizon.",
                context={"covariate": col, "missing": missing},
            )
    return merged


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
    merged = future_index.merge(
        future_x,
        on=[uid_col, ds_col],
        how="left",
    )
    covariate_cols = [c for c in future_x.columns if c not in {uid_col, ds_col}]
    for col in covariate_cols:
        if merged[col].isna().any():
            missing = int(merged[col].isna().sum())
            raise ECovariateIncompleteKnown(
                f"Future-known covariate '{col}' missing {missing} values in horizon.",
                context={"covariate": col, "missing": missing},
            )
    return future_x.copy()


__all__ = [
    "AlignedDataset",
    "CovariateBundle",
    "align_covariates",
]
