"""Tests for QA covariate leakage and policy handling."""

from __future__ import annotations

import pandas as pd

from tsagentkit import TaskSpec
from tsagentkit.qa import run_qa


def test_observed_covariate_leakage_detected() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, None],
            "promo": [0, 1, 1],
        }
    )
    spec = TaskSpec(horizon=1, freq="D", covariate_policy="observed")

    report = run_qa(df, spec, mode="standard")

    assert report.leakage_detected is True
    assert any(issue["type"] == "covariate_leakage" for issue in report.issues)


def test_known_covariates_do_not_trigger_leakage() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, None],
            "promo": [0, 1, 1],
        }
    )
    spec = TaskSpec(horizon=1, freq="D", covariate_policy="known")

    report = run_qa(df, spec, mode="standard")

    assert report.leakage_detected is False
    assert not any(issue["type"] == "covariate_leakage" for issue in report.issues)


def test_auto_covariate_inference_uses_future_values() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, None],
            "known_cov": [5, 6, 7],
            "observed_cov": [1.0, 2.0, None],
        }
    )
    spec = TaskSpec(horizon=1, freq="D", covariate_policy="auto")

    report = run_qa(df, spec, mode="standard")

    assert report.leakage_detected is False


def test_apply_repairs_interpolates_missing_values() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, None, 3.0],
        }
    )
    spec = TaskSpec(horizon=1, freq="D")

    report = run_qa(df, spec, mode="standard", apply_repairs=True)

    assert df["y"].isna().sum() == 0
    assert any(r["type"] == "missing_values" for r in report.repairs)


def test_apply_repairs_winsorizes_outliers() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [0.0, 0.0, 10.0],
        }
    )
    spec = TaskSpec(horizon=1, freq="D")

    report = run_qa(df, spec, mode="standard", apply_repairs=True, outlier_z=1.0)

    assert df["y"].max() < 10.0
    assert any(r["type"] == "outliers" for r in report.repairs)
