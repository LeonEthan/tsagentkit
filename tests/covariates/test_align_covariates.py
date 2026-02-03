"""Tests for covariate alignment policy enforcement."""

import pandas as pd
import pytest

from tsagentkit.covariates import align_covariates
from tsagentkit.contracts import (
    CovariateSpec,
    ECovariateIncompleteKnown,
    ETaskSpecInvalid,
    TaskSpec,
)


def _sample_panel() -> pd.DataFrame:
    return pd.DataFrame({
        "unique_id": ["A", "A", "A"],
        "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
        "y": [1.0, 2.0, None],
        "cov1": [10.0, 11.0, None],
        "cov2": [3.0, 4.0, 5.0],
    })


def test_spec_policy_requires_covariate_spec() -> None:
    panel = _sample_panel()
    spec = TaskSpec(h=1, freq="D", covariate_policy="spec")

    with pytest.raises(ETaskSpecInvalid, match="covariate_policy='spec'"):
        align_covariates(panel, spec)


def test_spec_policy_requires_roles_for_all_covariates() -> None:
    panel = _sample_panel()
    spec = TaskSpec(
        h=1,
        freq="D",
        covariate_policy="spec",
        covariates=CovariateSpec(roles={}),
    )

    with pytest.raises(ETaskSpecInvalid, match="requires explicit roles"):
        align_covariates(panel, spec)


def test_spec_policy_rejects_missing_columns() -> None:
    panel = _sample_panel()
    spec = TaskSpec(
        h=1,
        freq="D",
        covariate_policy="spec",
        covariates=CovariateSpec(roles={"cov1": "past", "cov3": "future_known"}),
    )

    with pytest.raises(ETaskSpecInvalid, match="not present in panel"):
        align_covariates(panel, spec)


def test_spec_policy_rejects_extra_panel_covariates() -> None:
    panel = _sample_panel()
    spec = TaskSpec(
        h=1,
        freq="D",
        covariate_policy="spec",
        covariates=CovariateSpec(roles={"cov1": "past"}),
    )

    with pytest.raises(ETaskSpecInvalid, match="requires roles for all panel covariates"):
        align_covariates(panel, spec)


def test_spec_policy_accepts_complete_roles() -> None:
    panel = _sample_panel()
    spec = TaskSpec(
        h=1,
        freq="D",
        covariate_policy="spec",
        covariates=CovariateSpec(roles={"cov1": "past", "cov2": "future_known"}),
    )

    aligned = align_covariates(panel, spec)
    assert aligned.past_x is not None
    assert aligned.future_x is not None


def test_known_policy_enforces_future_coverage() -> None:
    panel = _sample_panel()
    spec = TaskSpec(h=1, freq="D", covariate_policy="known")

    with pytest.raises(ECovariateIncompleteKnown, match="missing"):
        align_covariates(panel, spec)


def test_spec_policy_future_known_requires_coverage() -> None:
    panel = _sample_panel()
    spec = TaskSpec(
        h=1,
        freq="D",
        covariate_policy="spec",
        covariates=CovariateSpec(roles={"cov1": "future_known", "cov2": "future_known"}),
    )

    with pytest.raises(ECovariateIncompleteKnown, match="missing"):
        align_covariates(panel, spec)
