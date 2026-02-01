# tsagentkit Skill Documentation

## What
Agent-facing quick reference for using tsagentkit correctly, including guardrails,
core modules, and the end-to-end workflow.

## When
Use this when writing or reviewing forecasting scripts that must comply with
tsagentkit's temporal integrity and leakage prevention rules.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `task_spec`: `TaskSpec` (horizon, freq, optional quantiles/covariate policy/repair_strategy)
- Optional: custom `fit_func`, `predict_func`, `monitoring_config`

## Workflow
1. `validate_contract` to enforce schema and ordering.
2. `run_qa` (repairs optional in non-strict modes).
3. Build `TSDataset`.
4. `make_plan` selects model and fallback ladder.
5. `rolling_backtest` (standard/strict).
6. `fit` then `predict`.
7. `package_run` returns `RunArtifact` with provenance.

---

## Guardrails (Critical)

| Guardrail | Error Code | Prevention |
|---|---|---|
| No random splits | `E_SPLIT_RANDOM_FORBIDDEN` | Data must be sorted by (`unique_id`, `ds`) |
| No covariate leakage | `E_COVARIATE_LEAKAGE` | Future covariates cannot be used |
| Temporal ordering | `E_CONTRACT_UNSORTED` | Dates must increase within each series |

---

## Module Reference

### contracts
**What**: Validates input data schema and types.

**When**: Before any forecasting operation.

**Inputs**:
- `unique_id` (str), `ds` (datetime), `y` (numeric)

**Workflow**:
```python
from tsagentkit import validate_contract, TaskSpec

report = validate_contract(df)
if not report.valid:
    report.raise_if_errors()

spec = TaskSpec(
    horizon=7,
    freq="D",
    quantiles=[0.1, 0.5, 0.9],
    repair_strategy={
        "interpolate_missing": True,
        "winsorize_outliers": True,
        "missing_method": "linear",
        "outlier_z": 3.0,
    },
)
```

### series
**What**: Immutable dataset container with sparsity profiling.

**When**: After validation, before modeling.

**Inputs**: DataFrame + TaskSpec.

**Workflow**:
```python
from tsagentkit import TSDataset, build_dataset

dataset = build_dataset(df, spec)
print(dataset.n_series)
print(dataset.date_range)
```

### router
**What**: Selects models and fallbacks based on data characteristics.

**When**: After building dataset.

**Inputs**: `TSDataset`, `TaskSpec`.

**Workflow**:
```python
from tsagentkit import make_plan

plan = make_plan(dataset, spec)
print(plan.primary_model, plan.fallback_chain)
```

### backtest
**What**: Rolling window validation without random splits.

**When**: Standard/strict mode before final training.

**Inputs**: Dataset, plan, fit/predict functions.

**Workflow**:
```python
from tsagentkit import rolling_backtest

report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit_func,
    predict_func=predict_func,
)
print(report.summary())
```

### serving
**What**: Full pipeline execution with provenance.

**When**: End-to-end forecasting runs.

**Inputs**: DataFrame + TaskSpec (+ optional configs).

**Workflow**:
```python
from tsagentkit import run_forecast

result = run_forecast(df, spec, mode="standard")
print(result.summary())
```
