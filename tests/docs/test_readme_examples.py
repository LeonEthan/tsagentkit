"""Smoke tests for README-style examples."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tsagentkit import (
    TaskSpec,
    align_covariates,
    build_dataset,
    make_plan,
    package_run,
    run_qa,
    run_forecast,
    validate_contract,
)
from tsagentkit.contracts import ForecastResult, ModelArtifact, Provenance
from tsagentkit.series import TSDataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_panel(n_points: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return pd.DataFrame(
        {
            "unique_id": ["A"] * n_points + ["B"] * n_points,
            "ds": list(dates) * 2,
            "y": list(range(n_points)) + list(range(n_points, 0, -1)),
        }
    )


def _fit_stub(dataset: TSDataset, plan) -> ModelArtifact:
    model_name = plan.candidate_models[0] if plan.candidate_models else "model"
    return ModelArtifact(model={"fitted": True}, model_name=model_name)


def _predict_stub(dataset: TSDataset, artifact: ModelArtifact, spec: TaskSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    step = pd.tseries.frequencies.to_offset(spec.freq or "D")
    last_dates = dataset.df.groupby("unique_id")["ds"].max()
    for uid, last_date in last_dates.items():
        for h in range(1, spec.horizon + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": last_date + h * step,
                    "model": artifact.model_name,
                    "yhat": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_readme_make_plan_tuple_usage() -> None:
    df = _make_panel()
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})
    dataset = TSDataset.from_dataframe(df, spec)

    plan, route_decision = make_plan(dataset, spec, use_tsfm=False)

    assert len(plan.candidate_models) > 0
    assert isinstance(route_decision.buckets, list)
    assert isinstance(route_decision.reasons, list)


def test_readme_assembly_first_package_run_usage() -> None:
    df = _make_panel()
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})

    validation = validate_contract(df)
    validation.raise_if_errors()
    qa_report = run_qa(df, spec, mode="quick")
    aligned = align_covariates(df, spec)
    dataset = build_dataset(aligned.panel, spec, validate=False).with_covariates(
        aligned,
        panel_with_covariates=df,
    )
    plan, route_decision = make_plan(dataset, spec, use_tsfm=False)
    model_artifact = _fit_stub(dataset, plan)
    forecast = ForecastResult(
        df=_predict_stub(dataset, model_artifact, spec),
        provenance=Provenance(
            run_id="readme-assembly-smoke",
            timestamp=datetime.now(UTC).isoformat(),
            data_signature="data_sig",
            task_signature="task_sig",
            plan_signature="plan_sig",
            model_signature=model_artifact.signature,
        ),
        model_name=model_artifact.model_name,
        horizon=spec.horizon,
    )
    result = package_run(
        forecast=forecast,
        plan=plan,
        task_spec=spec.model_dump(),
        qa_report=qa_report,
        model_artifact=model_artifact,
        provenance=forecast.provenance,
        metadata={"route_decision": route_decision.model_dump()},
    )

    assert not result.forecast.df.empty
    assert isinstance(result.forecast.model_name, str)
    assert isinstance(result.provenance.data_signature, str)
    assert result.provenance.data_signature


def test_readme_run_forecast_wrapper_accessors() -> None:
    df = _make_panel()
    spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})

    result = run_forecast(
        df,
        spec,
        mode="quick",
        fit_func=_fit_stub,
        predict_func=_predict_stub,
    )

    assert not result.forecast.df.empty
    assert isinstance(result.forecast.model_name, str)
    assert isinstance(result.provenance.data_signature, str)
    assert result.provenance.data_signature


def test_readme_assembly_first_headings_present() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    docs_readme = (_repo_root() / "docs" / "README.md").read_text(encoding="utf-8")

    assert "### Assembly-First (Recommended)" in readme
    assert "### Convenience Wrapper (`run_forecast`)" in readme
    assert readme.index("### Assembly-First (Recommended)") < readme.index(
        "### Convenience Wrapper (`run_forecast`)"
    )
    assert "package_run" in docs_readme


def test_architecture_doc_assembly_first_consistency() -> None:
    architecture = (_repo_root() / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "assembly-first" in architecture
    assert "`package_run()`" in architecture
    assert "`run_forecast()` (convenience wrapper" in architecture
