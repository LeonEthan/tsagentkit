"""Tests for sktime adapter integration."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.contracts import TaskSpec
from tsagentkit.covariates import align_covariates
from tsagentkit.router import PlanSpec
from tsagentkit.series import TSDataset


def test_sktime_adapter_with_future_covariates() -> None:
    sktime = pytest.importorskip("sktime")
    assert sktime is not None

    df_hist = pd.DataFrame({
        "unique_id": ["A"] * 10,
        "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
        "y": [float(v) for v in range(10)],
        "promo": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    df_future = pd.DataFrame({
        "unique_id": ["A"] * 2,
        "ds": pd.date_range("2024-01-11", periods=2, freq="D"),
        "y": [np.nan, np.nan],
        "promo": [1, 0],
    })
    panel = pd.concat([df_hist, df_future], ignore_index=True)

    spec = TaskSpec(h=2, freq="D", covariate_policy="known")
    aligned = align_covariates(panel, spec)

    dataset = TSDataset.from_dataframe(df_hist, spec, validate=False)
    dataset = dataset.with_covariates(
        aligned,
        panel_with_covariates=panel,
    )

    plan = PlanSpec(plan_name="default", candidate_models=["sktime-naive"])

    from tsagentkit.models.sktime import fit_sktime, predict_sktime

    artifact = fit_sktime("sktime-naive", dataset, plan, covariates=aligned)
    result = predict_sktime(dataset, artifact, spec, covariates=aligned)

    assert len(result.df) == 2
    assert "model" in result.df.columns
    assert result.model_name == "sktime-naive"
