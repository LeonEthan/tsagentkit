"""Smoke tests and content guards for skill documentation examples."""

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
    run_forecast,
    run_qa,
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


def test_skill_docs_taskspec_quantiles_alias() -> None:
    spec = TaskSpec(
        h=7,
        freq="D",
        quantiles=[0.1, 0.5, 0.9],
        tsfm_policy={"mode": "preferred"},
    )
    assert spec.quantiles == [0.1, 0.5, 0.9]


def test_skill_docs_assembly_first_and_wrapper_accessors() -> None:
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
    assert len(plan.candidate_models) > 0
    assert isinstance(route_decision.reasons, list)

    model_artifact = _fit_stub(dataset, plan)
    forecast = ForecastResult(
        df=_predict_stub(dataset, model_artifact, spec),
        provenance=Provenance(
            run_id="skill-assembly-smoke",
            timestamp=datetime.now(UTC).isoformat(),
            data_signature="data_sig",
            task_signature="task_sig",
            plan_signature="plan_sig",
            model_signature=model_artifact.signature,
        ),
        model_name=model_artifact.model_name,
        horizon=spec.horizon,
    )
    assembled = package_run(
        forecast=forecast,
        plan=plan,
        task_spec=spec.model_dump(),
        qa_report=qa_report,
        model_artifact=model_artifact,
        provenance=forecast.provenance,
        metadata={"route_decision": route_decision.model_dump()},
    )

    assert not assembled.forecast.df.empty
    assert isinstance(assembled.forecast.model_name, str)
    assert isinstance(assembled.provenance.data_signature, str)

    result = run_forecast(
        df,
        spec,
        mode="quick",
        fit_func=_fit_stub,
        predict_func=_predict_stub,
    )

    # Access patterns used in skill docs.
    assert not result.forecast.df.head().empty
    assert isinstance(result.forecast.model_name, str)
    assert isinstance(result.provenance.data_signature, str)
    assert isinstance(result.provenance.timestamp, str)


def test_skill_docs_do_not_use_stale_patterns() -> None:
    files = [
        _repo_root() / "skill" / "README.md",
        _repo_root() / "skill" / "tool_map.md",
        _repo_root() / "skill" / "recipes.md",
        _repo_root() / "src" / "tsagentkit" / "skill" / "README.md",
        _repo_root() / "src" / "tsagentkit" / "skill" / "recipes.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = [
        "plan.primary_model",
        "plan.fallback_chain",
        "result.model_name",
        "result.forecast.head()",
        "provenance['data_signature']",
        "E_MODEL_FIT_FAILED",
        "E_CONTRACT_UNSORTED",
        "docs/recipes/RECIPE_TROUBLESHOOTING.md",
    ]
    for pattern in forbidden:
        assert pattern not in text


def test_skill_docs_assembly_first_headings_present() -> None:
    readme = (_repo_root() / "skill" / "README.md").read_text(encoding="utf-8")
    assert "Pattern 1: Assembly-First Pipeline (Recommended)" in readme
    assert "Pattern 2: Quick Forecast Wrapper (Convenience)" in readme
    assert readme.index("Pattern 1: Assembly-First Pipeline (Recommended)") < readme.index(
        "Pattern 2: Quick Forecast Wrapper (Convenience)"
    )


def test_skill_docs_repair_examples_match_api_signature() -> None:
    files = [
        _repo_root() / "skill" / "QUICKSTART.md",
        _repo_root() / "skill" / "TROUBLESHOOTING.md",
        _repo_root() / "src" / "tsagentkit" / "skill" / "QUICKSTART.md",
        _repo_root() / "src" / "tsagentkit" / "skill" / "TROUBLESHOOTING.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "repair(df, e)" not in text
    assert "repair(df)" in text
