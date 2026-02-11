"""Tests for notebook-style local score computations."""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit.gift_eval import score as score_mod


def test_compute_aggregate_scores_returns_expected_means() -> None:
    df = pd.DataFrame(
        {
            "dataset": ["a/H/short", "b/H/short"],
            "eval_metrics/MASE[0.5]": [1.0, 3.0],
            "eval_metrics/mean_weighted_sum_quantile_loss": [2.0, 4.0],
            "eval_metrics/sMAPE[0.5]": [10.0, 30.0],
        }
    )

    scores = score_mod.compute_aggregate_scores(df)
    assert scores["rows"] == 2.0
    assert scores["mase_mean"] == 2.0
    assert scores["crps_mean"] == 3.0
    assert scores["smape_mean"] == 20.0


def test_compute_normalized_scores_aligns_by_dataset_key() -> None:
    results_df = pd.DataFrame(
        {
            "dataset": ["b/H/short", "a/H/short"],
            "eval_metrics/MASE[0.5]": [2.0, 1.0],
            "eval_metrics/mean_weighted_sum_quantile_loss": [8.0, 2.0],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "dataset": ["a/H/short", "b/H/short"],
            "eval_metrics/MASE[0.5]": [0.5, 1.0],
            "eval_metrics/mean_weighted_sum_quantile_loss": [1.0, 4.0],
        }
    )

    normalized = score_mod.compute_normalized_scores(results_df, baseline_df)
    # Ratios: MASE [2.0, 2.0], CRPS [2.0, 2.0] -> geometric means both 2.0
    assert normalized["normalized_mase_geo"] == 2.0
    assert normalized["normalized_crps_geo"] == 2.0


def test_compute_normalized_scores_requires_complete_baseline() -> None:
    results_df = pd.DataFrame(
        {
            "dataset": ["a/H/short", "missing/H/short"],
            "eval_metrics/MASE[0.5]": [1.0, 2.0],
            "eval_metrics/mean_weighted_sum_quantile_loss": [1.0, 2.0],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "dataset": ["a/H/short"],
            "eval_metrics/MASE[0.5]": [1.0],
            "eval_metrics/mean_weighted_sum_quantile_loss": [1.0],
        }
    )

    with pytest.raises(ValueError, match="Baseline is missing"):
        score_mod.compute_normalized_scores(results_df, baseline_df)
