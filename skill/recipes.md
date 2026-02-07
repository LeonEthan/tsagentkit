# tsagentkit Recipes

## What
Runnable templates for common production forecasting tasks with assembly-first composition.

## When
Use these recipes when an agent needs a correct baseline flow that preserves temporal
integrity and can be extended for production controls.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `task_spec`: `TaskSpec`
- Optional: covariates, hierarchy, custom model functions, artifact storage path

## Workflow
- Prefer `validate_contract -> run_qa -> align_covariates -> build_dataset -> make_plan -> fit/predict -> package_run`.
- Use `run_forecast` only when orchestration convenience is desired.

---

## Recipe 1: Retail Daily Sales (Assembly-First)
```python
import pandas as pd
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

spec = TaskSpec(h=14, freq="D", quantiles=[0.1, 0.5, 0.9])
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

## Recipe 2: TSFM Required Policy (No Silent Downgrade)
```python
from tsagentkit import TaskSpec, build_dataset, make_plan

spec = TaskSpec(
    h=7,
    freq="D",
    tsfm_policy={
        "mode": "required",
        "adapters": ["chronos", "moirai", "timesfm"],
    },
)
dataset = build_dataset(df, spec)
plan, decision = make_plan(dataset, spec)
print(plan.candidate_models)
print(decision.reasons)
```

## Recipe 3: Agent-Oriented Plan Graph + Capabilities
```python
from tsagentkit import (
    attach_plan_graph,
    build_dataset,
    get_adapter_capability,
    list_adapter_capabilities,
    make_plan,
)

spec = TaskSpec(h=14, freq="D")
dataset = build_dataset(df, spec)
plan, _decision = make_plan(dataset, spec)
plan = attach_plan_graph(plan, include_backtest=True)

capabilities = list_adapter_capabilities()
chronos = get_adapter_capability("chronos")
print(plan.graph.entrypoints)
print(chronos.available, chronos.supports_future_covariates)
```

## Recipe 4: Lifecycle Save/Load/Validate/Replay
```python
from tsagentkit import (
    load_run_artifact,
    replay_forecast_from_artifact,
    save_run_artifact,
    validate_run_artifact_for_serving,
)

save_run_artifact(run_artifact, "artifacts/run_artifact.json")
loaded = load_run_artifact("artifacts/run_artifact.json")
compat = validate_run_artifact_for_serving(
    loaded,
    expected_task_signature=loaded.provenance.task_signature,
    expected_plan_signature=loaded.provenance.plan_signature,
)
replayed = replay_forecast_from_artifact(loaded)
print(compat["artifact_schema_version"], replayed.df.shape)
```

## Recipe 5: Wrapper Path for Fast Iteration
```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec(h=24, freq="H")
result = run_forecast(df, spec, mode="standard")
print(result.summary())
```

## Recipe 6: Robust Error Handling
```python
from tsagentkit import TaskSpec, run_forecast, validate_contract
from tsagentkit.contracts import EContractMissingColumn, ESplitRandomForbidden, TSAgentKitError

spec = TaskSpec(h=7, freq="D")
try:
    report = validate_contract(df)
    report.raise_if_errors()
    result = run_forecast(df, spec, mode="standard")
except EContractMissingColumn:
    # ensure unique_id/ds/y exist
    raise
except ESplitRandomForbidden:
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    result = run_forecast(df, spec, mode="standard")
except TSAgentKitError:
    raise
```

## Notes
- Keep all training/evaluation splits temporal.
- Treat TSFM adapters as primary path unless explicit policy disables them.
- Persist artifacts before deployment and validate signatures at serving boundary.
