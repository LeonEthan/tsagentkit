"""Submission contract utilities for GIFT-Eval leaderboard packaging."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_RESULT_COLUMNS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]

MANDATORY_METRIC_COLUMNS = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]

ALLOWED_MODEL_TYPES = {
    "statistical",
    "deep-learning",
    "agentic",
    "pretrained",
    "fine-tuned",
    "zero-shot",
}
YES_NO_VALUES = {"Yes", "No"}
# Current benchmark matrix:
# 55 short datasets + (21 medium/long datasets * 2 terms) = 97 rows.
DEFAULT_EXPECTED_ROWS = 97


class SubmissionValidationError(ValueError):
    """Raised when submission artifacts fail validation."""


def _require_non_empty(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise SubmissionValidationError(f"Field '{name}' must be non-empty.")
    return text


def validate_results_csv(
    results_file: Path | str,
    *,
    require_full_rows: bool = True,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
) -> pd.DataFrame:
    """Validate `all_results.csv` against expected leaderboard schema."""
    path = Path(results_file)
    if not path.exists():
        raise SubmissionValidationError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_RESULT_COLUMNS:
        raise SubmissionValidationError(
            "Invalid result columns. Expected exactly:\n"
            + ", ".join(EXPECTED_RESULT_COLUMNS)
        )

    if require_full_rows and len(df) != expected_rows:
        raise SubmissionValidationError(
            f"Expected {expected_rows} rows for full benchmark, found {len(df)}."
        )
    if not require_full_rows and len(df) == 0:
        raise SubmissionValidationError("Results must contain at least one row.")

    if df["dataset"].duplicated().any():
        duplicates = df.loc[df["dataset"].duplicated(), "dataset"].tolist()
        raise SubmissionValidationError(
            f"Duplicate dataset rows detected: {duplicates[:5]}"
        )

    if df["model"].isna().any() or (df["model"].astype(str).str.strip() == "").any():
        raise SubmissionValidationError("Column 'model' contains empty values.")

    if df["domain"].isna().any() or (df["domain"].astype(str).str.strip() == "").any():
        raise SubmissionValidationError("Column 'domain' contains empty values.")

    for col in MANDATORY_METRIC_COLUMNS:
        series = df[col]
        if series.isna().any():
            raise SubmissionValidationError(f"Metric column '{col}' contains NaN values.")
        finite = series.map(lambda x: math.isfinite(float(x)))
        if not finite.all():
            raise SubmissionValidationError(f"Metric column '{col}' contains non-finite values.")

    return df


def build_config_payload(
    *,
    model_name: str,
    model_type: str,
    model_dtype: str,
    model_link: str,
    code_link: str,
    org: str,
    testdata_leakage: str,
    replication_code_available: str,
) -> dict[str, str]:
    """Build and validate `config.json` payload for leaderboard submission."""
    model_name = _require_non_empty("model_name", model_name)
    model_type = _require_non_empty("model_type", model_type)
    model_dtype = _require_non_empty("model_dtype", model_dtype)
    model_link = _require_non_empty("model_link", model_link)
    code_link = _require_non_empty("code_link", code_link)
    org = _require_non_empty("org", org)
    testdata_leakage = _require_non_empty("testdata_leakage", testdata_leakage)
    replication_code_available = _require_non_empty(
        "replication_code_available",
        replication_code_available,
    )

    if model_type not in ALLOWED_MODEL_TYPES:
        raise SubmissionValidationError(
            f"Invalid model_type '{model_type}'. Allowed: {sorted(ALLOWED_MODEL_TYPES)}"
        )
    if testdata_leakage not in YES_NO_VALUES:
        raise SubmissionValidationError("testdata_leakage must be one of {'Yes', 'No'}.")
    if replication_code_available not in YES_NO_VALUES:
        raise SubmissionValidationError(
            "replication_code_available must be one of {'Yes', 'No'}."
        )

    return {
        "model": model_name,
        "model_type": model_type,
        "model_dtype": model_dtype,
        "model_link": model_link,
        "code_link": code_link,
        "org": org,
        "testdata_leakage": testdata_leakage,
        "replication_code_available": replication_code_available,
    }


def prepare_submission(
    *,
    results_file: Path | str,
    output_path: Path | str,
    config_payload: dict[str, Any],
    require_full_rows: bool = True,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create `submissions/<model>/all_results.csv` and `config.json`."""
    results_path = Path(results_file)
    output_root = Path(output_path)
    model_name = _require_non_empty("config_payload.model", str(config_payload.get("model", "")))

    validate_results_csv(
        results_path,
        require_full_rows=require_full_rows,
        expected_rows=expected_rows,
    )

    submission_dir = output_root / model_name
    if submission_dir.exists() and not overwrite:
        raise SubmissionValidationError(
            f"Submission directory already exists: {submission_dir}. Use --overwrite to replace."
        )
    submission_dir.mkdir(parents=True, exist_ok=True)

    csv_target = submission_dir / "all_results.csv"
    cfg_target = submission_dir / "config.json"
    shutil.copy2(results_path, csv_target)
    cfg_target.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    return csv_target, cfg_target
