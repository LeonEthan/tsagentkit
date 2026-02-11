"""Scoring helpers for GIFT-Eval result files."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

MASE_COL = "eval_metrics/MASE[0.5]"
CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"
SMAPE_COL = "eval_metrics/sMAPE[0.5]"
DATASET_COL = "dataset"


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def geometric_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute geometric mean for empty values.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Geometric mean input contains non-finite values.")
    if np.any(arr <= 0):
        raise ValueError("Geometric mean input must be positive.")
    return float(np.exp(np.mean(np.log(arr))))


def compute_aggregate_scores(results_df: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate local scores from all_results rows."""
    _require_columns(results_df, [MASE_COL, CRPS_COL, SMAPE_COL, DATASET_COL], "results_df")
    return {
        "rows": float(len(results_df)),
        "mase_mean": float(results_df[MASE_COL].mean()),
        "mase_median": float(results_df[MASE_COL].median()),
        "crps_mean": float(results_df[CRPS_COL].mean()),
        "crps_median": float(results_df[CRPS_COL].median()),
        "smape_mean": float(results_df[SMAPE_COL].mean()),
        "smape_median": float(results_df[SMAPE_COL].median()),
    }


def compute_normalized_scores(results_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict[str, float]:
    """Compute notebook-style normalized geometric scores against a baseline."""
    _require_columns(results_df, [DATASET_COL, MASE_COL, CRPS_COL], "results_df")
    _require_columns(baseline_df, [DATASET_COL, MASE_COL, CRPS_COL], "baseline_df")

    baseline_by_dataset = baseline_df.drop_duplicates(subset=[DATASET_COL]).set_index(DATASET_COL)
    missing = sorted(set(results_df[DATASET_COL]) - set(baseline_by_dataset.index))
    if missing:
        raise ValueError(f"Baseline is missing {len(missing)} dataset(s): {missing[:5]}")

    aligned_baseline = baseline_by_dataset.loc[results_df[DATASET_COL]]
    mase_ratio = results_df[MASE_COL].to_numpy(dtype=float) / aligned_baseline[MASE_COL].to_numpy(dtype=float)
    crps_ratio = results_df[CRPS_COL].to_numpy(dtype=float) / aligned_baseline[CRPS_COL].to_numpy(dtype=float)

    if np.any(~np.isfinite(mase_ratio)) or np.any(~np.isfinite(crps_ratio)):
        raise ValueError("Normalized scores contain non-finite values.")
    if np.any(mase_ratio <= 0) or np.any(crps_ratio <= 0):
        raise ValueError("Normalized scores must be positive for geometric mean.")

    return {
        "normalized_mase_geo": geometric_mean(mase_ratio),
        "normalized_crps_geo": geometric_mean(crps_ratio),
    }
