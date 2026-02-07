# tsagentkit Skill Documentation

## What
Agent-facing guide for building production time-series forecasting systems with `tsagentkit`.
This skill is assembly-first and TSFM-first: coding agents should compose explicit steps,
while `run_forecast` remains a convenience wrapper.

## When
Use this guide when an agent needs to:
- validate and repair panel data,
- route to TSFM/baseline candidates deterministically,
- assemble forecasting steps with explicit artifacts,
- persist and replay production artifacts safely.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `task_spec`: `TaskSpec`
- Optional: covariates, hierarchy, monitoring config
- Optional: custom `fit_func` and `predict_func`

## Workflow
1. `validate_contract`
2. `run_qa`
3. `align_covariates`
4. `build_dataset`
5. `make_plan`
6. `fit` and `predict`
7. `package_run`
8. Optional lifecycle: `save_run_artifact` -> `load_run_artifact` -> `validate_run_artifact_for_serving` -> `replay_forecast_from_artifact`

---

## Core Principles
- Assembly-first by default. Prefer explicit step composition over monolithic wrappers.
- TSFM-first routing. Chronos/Moirai/TimesFM are first-class and policy-driven.
- Deterministic provenance. Keep signatures, decisions, and fallback events auditable.
- Guardrails always on. Never use random splits or future leakage.

## Guardrails
- `E_SPLIT_RANDOM_FORBIDDEN`: never randomize temporal order.
- `E_DS_NOT_MONOTONIC`: sort by `unique_id`, `ds` before processing.
- `E_COVARIATE_LEAKAGE`: do not feed future-observed covariates.
- `E_MODEL_FIT_FAIL` / `E_MODEL_PREDICT_FAIL`: model execution failure.
- `E_FALLBACK_EXHAUSTED`: all fallback candidates failed.
- `E_ARTIFACT_SCHEMA_INCOMPATIBLE`: artifact schema/type mismatch at load/serve boundary.
- `E_ARTIFACT_LOAD_FAILED`: artifact payload cannot be safely reconstructed.

## Module Map
- `contracts`: task specs, payload contracts, structured errors
- `series`: immutable `TSDataset`, sparsity profiling
- `qa`: quality checks and optional repairs
- `router`: deterministic candidate selection and fallback policy
- `models`: `fit`/`predict`, TSFM adapter capability APIs
- `backtest`: rolling temporal validation
- `serving`: packaging + lifecycle save/load/validate/replay

## Pattern 1: Assembly-First Pipeline (Recommended)
```python
from tsagentkit import (
    TaskSpec,
    align_covariates,
    build_dataset,
    fit,
    make_plan,
    package_run,
    predict,
    run_qa,
    validate_contract,
)

spec = TaskSpec(h=7, freq="D")
report = validate_contract(df)
report.raise_if_errors()
qa_report = run_qa(df, spec, mode="standard")
aligned = align_covariates(df, spec)
dataset = build_dataset(aligned.panel, spec, validate=False).with_covariates(
    aligned,
    panel_with_covariates=df,
)
plan, route_decision = make_plan(dataset, spec)
model_artifact = fit(dataset, plan, covariates=aligned)
forecast = predict(dataset, model_artifact, spec, covariates=aligned)
run_artifact = package_run(
    forecast=forecast,
    plan=plan,
    task_spec=spec.model_dump(),
    qa_report=qa_report,
    model_artifact=model_artifact,
    provenance=forecast.provenance,
    metadata={"route_decision": route_decision.model_dump()},
)
```

## Pattern 2: Quick Forecast Wrapper (Convenience)
```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec(h=7, freq="D")
result = run_forecast(df, spec, mode="quick")
forecast_df = result.forecast.df
```

## Pattern 3: Agent Graph + TSFM Capability Gate
```python
from tsagentkit import (
    TaskSpec,
    attach_plan_graph,
    build_dataset,
    get_adapter_capability,
    list_adapter_capabilities,
    make_plan,
)

spec = TaskSpec(h=14, freq="D", tsfm_policy={"mode": "required"})
dataset = build_dataset(df, spec)
plan, _decision = make_plan(dataset, spec)
plan = attach_plan_graph(plan, include_backtest=True)

capabilities = list_adapter_capabilities()
chronos = get_adapter_capability("chronos")
print(plan.graph.entrypoints, chronos.available)
```

## Pattern 4: Artifact Lifecycle (Phase 5)
```python
from tsagentkit import (
    load_run_artifact,
    replay_forecast_from_artifact,
    save_run_artifact,
    validate_run_artifact_for_serving,
)

save_run_artifact(run_artifact, "artifacts/run_artifact.json")
loaded = load_run_artifact("artifacts/run_artifact.json")
validate_run_artifact_for_serving(
    loaded,
    expected_task_signature=loaded.provenance.task_signature,
    expected_plan_signature=loaded.provenance.plan_signature,
)
replayed = replay_forecast_from_artifact(loaded)
```

## Data Requirements
- Required columns: `unique_id`, `ds`, `y`
- `ds` must be datetime-like and monotonic per series
- duplicates on (`unique_id`, `ds`) are invalid

## Next Steps
- Use `skill/recipes.md` for end-to-end templates.
- Use `skill/tool_map.md` for task-to-API lookup.
- Use `docs/API_STABILITY.md` for compatibility boundaries.
