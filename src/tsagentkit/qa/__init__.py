"""QA checks and PIT-safe repairs for tsagentkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from tsagentkit.contracts import (
    ECovariateIncompleteKnown,
    ECovariateLeakage,
    ECovariateStaticInvalid,
    EQARepairPeeksFuture,
    RepairReport,
    TaskSpec,
)
from tsagentkit.covariates import align_covariates


@dataclass(frozen=True)
class QAReport:
    """Quality assurance report."""

    issues: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[RepairReport] = field(default_factory=list)
    leakage_detected: bool = False

    def has_critical_issues(self) -> bool:
        return any(issue.get("severity") == "critical" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        repairs_list: list[dict[str, Any]] = []
        for r in self.repairs:
            if hasattr(r, "to_dict"):
                repairs_list.append(r.to_dict())
            else:
                repairs_list.append(r)
        return {
            "issues": self.issues,
            "repairs": repairs_list,
            "leakage_detected": self.leakage_detected,
        }


def run_qa(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    mode: Literal["quick", "standard", "strict"] = "standard",
    zero_threshold: float = 0.3,
    outlier_z: float = 3.0,
    apply_repairs: bool = False,
    repair_strategy: dict[str, Any] | None = None,
    skip_covariate_checks: bool = False,
) -> QAReport:
    """Run QA checks for missing values, gaps, outliers, and leakage."""
    repair_strategy = repair_strategy or {}
    missing_method = repair_strategy.get("missing_method", "ffill")
    winsorize_cfg = repair_strategy.get("winsorize", {"window": 30, "lower_q": 0.01, "upper_q": 0.99})
    median_cfg = repair_strategy.get("median_filter", {"window": 7})
    outlier_z = float(repair_strategy.get("outlier_z", outlier_z))

    issues: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    leakage_detected = False

    contract = task_spec.panel_contract
    uid_col = contract.unique_id_col
    ds_col = contract.ds_col
    y_col = contract.y_col

    df = data
    if not pd.api.types.is_datetime64_any_dtype(df[ds_col]):
        df[ds_col] = pd.to_datetime(df[ds_col])

    # Per-series last observed
    last_observed = (
        df[df[y_col].notna()]
        .groupby(uid_col)[ds_col]
        .max()
        .to_dict()
    )

    # Missing values in observed history only
    missing_mask = df[y_col].isna()
    if last_observed:
        mask = df[uid_col].map(last_observed)
        missing_mask = missing_mask & (df[ds_col] <= mask)
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        issues.append(
            {
                "type": "missing_values",
                "column": y_col,
                "count": missing_count,
                "severity": "critical" if mode == "strict" else "warning",
            }
        )

    # Gaps per series
    gap_count = 0
    gap_ratio = 0.0
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid].sort_values(ds_col)
        if series.empty:
            continue
        full_range = pd.date_range(
            start=series[ds_col].min(),
            end=series[ds_col].max(),
            freq=task_spec.freq,
        )
        missing = len(full_range) - len(series)
        if missing > 0:
            gap_count += missing
            gap_ratio += missing / max(len(full_range), 1)

    if gap_count > 0:
        issues.append(
            {
                "type": "gaps",
                "count": gap_count,
                "ratio": gap_ratio / max(df[uid_col].nunique(), 1),
                "severity": "warning",
            }
        )

    # Zero density
    zero_ratio = float(np.mean(df[y_col] == 0)) if len(df) > 0 else 0.0
    if zero_ratio > zero_threshold:
        issues.append(
            {
                "type": "zero_density",
                "ratio": zero_ratio,
                "threshold": zero_threshold,
                "severity": "warning",
            }
        )

    # Outliers (z-score per series)
    outlier_count = 0
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid][y_col].astype(float)
        if series.empty:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            continue
        z_scores = (series - mean) / std
        outlier_count += int((np.abs(z_scores) > outlier_z).sum())

    if outlier_count > 0:
        issues.append(
            {
                "type": "outliers",
                "count": outlier_count,
                "z_threshold": outlier_z,
                "severity": "warning",
            }
        )

    # Monotonicity check per series
    monotonic_violations = 0
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid]
        if not series[ds_col].is_monotonic_increasing:
            monotonic_violations += 1
    if monotonic_violations > 0:
        issues.append(
            {
                "type": "ds_not_monotonic",
                "count": monotonic_violations,
                "severity": "critical" if mode == "strict" else "warning",
            }
        )

    # Minimum history length check
    min_history = task_spec.backtest.min_train_size
    if min_history:
        lengths = df[df[y_col].notna()].groupby(uid_col).size()
        short = lengths[lengths < min_history]
        if not short.empty:
            issues.append(
                {
                    "type": "min_history",
                    "count": int(short.shape[0]),
                    "min_train_size": min_history,
                    "severity": "critical" if mode == "strict" else "warning",
                }
            )

    # Covariate guardrails
    if not skip_covariate_checks:
        try:
            align_covariates(df, task_spec)
        except (ECovariateLeakage, ECovariateIncompleteKnown, ECovariateStaticInvalid) as exc:
            leakage_detected = isinstance(exc, ECovariateLeakage)
            issues.append(
                {
                    "type": "covariate_guardrail",
                    "error": str(exc),
                    "severity": "critical",
                }
            )
            raise

    repairs: list[RepairReport] = []
    if apply_repairs:
        repairs = _apply_repairs(
            df,
            uid_col=uid_col,
            ds_col=ds_col,
            y_col=y_col,
            last_observed=last_observed,
            missing_method=missing_method,
            winsorize_cfg=winsorize_cfg,
            median_cfg=median_cfg,
            strict=(mode == "strict"),
        )

    return QAReport(
        issues=issues,
        repairs=repairs,
        leakage_detected=leakage_detected,
    )


def _apply_repairs(
    data: pd.DataFrame,
    uid_col: str,
    ds_col: str,
    y_col: str,
    last_observed: dict[str, Any],
    missing_method: str,
    winsorize_cfg: dict[str, Any],
    median_cfg: dict[str, Any],
    strict: bool,
) -> list[RepairReport]:
    if y_col in data.columns:
        data[y_col] = data[y_col].astype(float)

    repairs: list[RepairReport] = []
    missing_filled = 0
    outliers_clipped = 0
    median_applied = 0

    # Track before/after statistics for audit trail
    before_stats: dict[str, Any] = {}
    after_stats: dict[str, Any] = {}

    for uid in data[uid_col].unique():
        series_idx = data[uid_col] == uid
        series = data.loc[series_idx].sort_values(ds_col).copy()
        if series.empty or not series[y_col].notna().any():
            continue

        last_obs = last_observed.get(uid)
        observed_mask = series[ds_col] <= last_obs if last_obs is not None else pd.Series(False, index=series.index)

        if missing_method in {"ffill", "bfill"}:
            if missing_method == "bfill" and strict:
                raise EQARepairPeeksFuture(
                    "bfill is non-causal in strict mode.",
                    context={"missing_method": missing_method},
                )
            missing_mask = series[y_col].isna() & observed_mask
            if missing_mask.any():
                if missing_method == "ffill":
                    filled = series.loc[observed_mask, y_col].ffill()
                else:
                    filled = series.loc[observed_mask, y_col].bfill()
                series.loc[observed_mask, y_col] = filled
                missing_filled += int(missing_mask.sum())

        # Winsorize using rolling historical quantiles (left-closed window)
        if winsorize_cfg:
            window = int(winsorize_cfg.get("window", 30))
            lower_q = float(winsorize_cfg.get("lower_q", 0.01))
            upper_q = float(winsorize_cfg.get("upper_q", 0.99))
            observed_values = series.loc[observed_mask, y_col].astype(float)
            shifted = observed_values.shift(1)
            lower = shifted.rolling(window, min_periods=1).quantile(lower_q)
            upper = shifted.rolling(window, min_periods=1).quantile(upper_q)
            clipped = observed_values.copy()
            clipped = clipped.where(lower.isna() | (clipped >= lower), lower)
            clipped = clipped.where(upper.isna() | (clipped <= upper), upper)
            outliers_clipped += int((clipped != observed_values).sum())
            series.loc[observed_mask, y_col] = clipped

        # Median filter using historical window (left-closed)
        if median_cfg:
            window = int(median_cfg.get("window", 7))
            observed_values = series.loc[observed_mask, y_col].astype(float)
            shifted = observed_values.shift(1)
            median = shifted.rolling(window, min_periods=1).median()
            filled = observed_values.where(median.isna(), median)
            median_applied += int((filled != observed_values).sum())
            series.loc[observed_mask, y_col] = filled

        data.loc[series.index, y_col] = series[y_col].values

    if missing_filled > 0:
        repairs.append(
            RepairReport(
                repair_type="missing_values",
                column=y_col,
                count=missing_filled,
                method=missing_method,
                scope="observed_history",
                pit_safe=missing_method != "bfill",
                validation_passed=True,
            )
        )

    if outliers_clipped > 0:
        repairs.append(
            RepairReport(
                repair_type="winsorize",
                column=y_col,
                count=outliers_clipped,
                method="rolling_quantiles",
                scope="observed_history",
                pit_safe=True,
                validation_passed=True,
            )
        )

    if median_applied > 0:
        repairs.append(
            RepairReport(
                repair_type="median_filter",
                column=y_col,
                count=median_applied,
                method="rolling_median",
                scope="observed_history",
                pit_safe=True,
                validation_passed=True,
            )
        )

    return repairs


__all__ = ["QAReport", "run_qa"]
