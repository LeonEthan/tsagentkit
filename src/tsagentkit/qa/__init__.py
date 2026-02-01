"""QA module stub for tsagentkit.

This is a minimal stub for v0.1. Full QA implementation in Phase 1.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from tsagentkit.contracts import ECovariateLeakage, TaskSpec
from tsagentkit.features.covariates import CovariateManager, infer_covariate_config


@dataclass(frozen=True)
class QAReport:
    """Quality assurance report.

    Placeholder for full QA implementation.
    """

    issues: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    leakage_detected: bool = False

    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.get("severity") == "critical" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "issues": self.issues,
            "repairs": self.repairs,
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
) -> QAReport:
    """Run QA checks for missing values, gaps, outliers, and leakage."""
    repair_strategy = repair_strategy or {}
    missing_method = repair_strategy.get("missing_method", "linear")
    interpolate_missing = repair_strategy.get("interpolate_missing", True)
    winsorize_outliers = repair_strategy.get("winsorize_outliers", True)
    outlier_z = float(repair_strategy.get("outlier_z", outlier_z))

    issues: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    leakage_detected = False

    # Identify future rows (y missing after last observed)
    last_observed = None
    if "y" in data.columns and data["y"].notna().any():
        last_observed = data.loc[data["y"].notna(), "ds"].max()
    future_mask = data["y"].isna()
    if last_observed is not None:
        future_mask = future_mask & (data["ds"] > last_observed)

    # Missing values in observed history only
    missing_mask = data["y"].isna()
    if last_observed is not None:
        missing_mask = missing_mask & (data["ds"] <= last_observed)
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        issues.append(
            {
                "type": "missing_values",
                "column": "y",
                "count": missing_count,
                "severity": "critical" if mode == "strict" else "warning",
            }
        )

    # Gaps
    gap_count = 0
    gap_ratio = 0.0
    for uid in data["unique_id"].unique():
        series = data[data["unique_id"] == uid].sort_values("ds")
        if series.empty:
            continue
        full_range = pd.date_range(
            start=series["ds"].min(),
            end=series["ds"].max(),
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
                "ratio": gap_ratio / max(data["unique_id"].nunique(), 1),
                "severity": "warning",
            }
        )

    # Zero density
    zero_ratio = float(np.mean(data["y"] == 0)) if len(data) > 0 else 0.0
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
    for uid in data["unique_id"].unique():
        series = data[data["unique_id"] == uid]["y"].astype(float)
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

    # Leakage detection for observed covariates
    covariate_config = infer_covariate_config(data, task_spec.covariate_policy)
    if covariate_config.known or covariate_config.observed:
        if covariate_config.known and future_mask.any():
            for col in covariate_config.known:
                missing_future = int(data.loc[future_mask, col].isna().sum())
                if missing_future > 0:
                    issues.append(
                        {
                            "type": "known_covariate_missing",
                            "column": col,
                            "count": missing_future,
                            "severity": "warning",
                        }
                    )

        if covariate_config.observed:
            manager = CovariateManager(
                known_covariates=covariate_config.known,
                observed_covariates=covariate_config.observed,
            )
            if future_mask.any():
                forecast_start = data.loc[future_mask, "ds"].min()
            else:
                forecast_start = data["ds"].max() + pd.tseries.frequencies.to_offset(
                    task_spec.freq
                )
            try:
                manager.validate_for_prediction(
                    data,
                    forecast_start=forecast_start,
                    horizon=task_spec.horizon,
                )
            except ECovariateLeakage as exc:
                leakage_detected = True
                issues.append(
                    {
                        "type": "covariate_leakage",
                        "columns": covariate_config.observed,
                        "severity": "critical" if mode == "strict" else "warning",
                        "error": str(exc),
                    }
                )

    if apply_repairs:
        repairs = _apply_repairs(
            data,
            missing_method=missing_method,
            interpolate_missing=interpolate_missing,
            winsorize_outliers=winsorize_outliers,
            outlier_z=outlier_z,
        )

    return QAReport(
        issues=issues,
        repairs=repairs,
        leakage_detected=leakage_detected,
    )


def _apply_repairs(
    data: pd.DataFrame,
    missing_method: str,
    interpolate_missing: bool,
    winsorize_outliers: bool,
    outlier_z: float,
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    missing_filled = 0
    outliers_clipped = 0

    if "y" not in data.columns:
        return repairs

    for uid in data["unique_id"].unique():
        series_idx = data["unique_id"] == uid
        series = data.loc[series_idx].sort_values("ds").copy()
        if series.empty or not series["y"].notna().any():
            continue

        last_observed = series.loc[series["y"].notna(), "ds"].max()
        observed_mask = series["ds"] <= last_observed

        if interpolate_missing:
            missing_mask = series["y"].isna() & observed_mask
            if missing_mask.any():
                series.loc[observed_mask, "y"] = (
                    series.loc[observed_mask, "y"]
                    .astype(float)
                    .interpolate(method=missing_method, limit_direction="both")
                )
                missing_filled += int(missing_mask.sum())

        if winsorize_outliers:
            observed_values = series.loc[observed_mask, "y"].astype(float)
            if observed_values.notna().any():
                mean = observed_values.mean()
                std = observed_values.std()
                if std and not np.isnan(std):
                    z_scores = (observed_values - mean) / std
                    out_mask = z_scores.abs() > outlier_z
                    if out_mask.any():
                        clipped = observed_values.clip(
                            lower=mean - outlier_z * std,
                            upper=mean + outlier_z * std,
                        )
                        series.loc[observed_mask, "y"] = clipped
                        outliers_clipped += int(out_mask.sum())

        data.loc[series.index, "y"] = series["y"].values

    if missing_filled > 0:
        repairs.append(
            {
                "type": "missing_values",
                "column": "y",
                "count": missing_filled,
                "method": missing_method,
                "scope": "observed_history",
            }
        )

    if outliers_clipped > 0:
        repairs.append(
            {
                "type": "outliers",
                "column": "y",
                "count": outliers_clipped,
                "method": "winsorize",
                "z_threshold": outlier_z,
            }
        )

    return repairs


__all__ = ["QAReport", "run_qa"]
