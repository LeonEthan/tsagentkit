# tsagentkit Quickstart

Get a forecast running in under 3 minutes.

---

## Installation

```bash
# Core (statistical baselines only)
pip install tsagentkit

# With TSFM adapters (Chronos, Moirai, TimesFM)
pip install tsagentkit[tsfm]

# With hierarchical reconciliation
pip install tsagentkit[hierarchy]

# With feature engineering (tsfeatures, tsfresh, sktime)
pip install tsagentkit[features]

# Everything
pip install tsagentkit[full]
```

## Minimal Example (2 lines)

```python
from tsagentkit import forecast

result = forecast(df, 7)
```

`forecast()` auto-detects column names, infers frequency, sorts data, and runs
the full pipeline. It accepts any DataFrame with time-series panel columns
mappable to `unique_id`, `ds`, `y` (common aliases like `date`, `value`,
`series_id` are recognized automatically).

## Standard Assembly-First Example (10 lines)

```python
from tsagentkit import (
    TaskSpec, validate_contract, run_qa,
    build_dataset, make_plan, fit, predict, package_run,
)

spec = TaskSpec(h=7, freq="D")
report = validate_contract(df)
report.raise_if_errors()
qa = run_qa(df, spec, mode="standard")
dataset = build_dataset(df, spec)
plan, decision = make_plan(dataset, spec)
model = fit(dataset, plan)
result = predict(dataset, model, spec)
artifact = package_run(forecast=result, plan=plan, task_spec=spec.model_dump(),
                       qa_report=qa, model_artifact=model, provenance=result.provenance)
```

## TaskSpec Presets

| Preset | TSFM Policy | Backtest Windows | Use Case |
|--------|------------|-----------------|----------|
| `TaskSpec.starter(h, freq)` | `preferred` (fallback OK) | 2 | Experimentation, quick iteration |
| `TaskSpec.production(h, freq)` | `required` (strict) | 5 | Production deployments |

```python
from tsagentkit import TaskSpec

# Quick experimentation — falls back to baselines if no TSFM installed
spec = TaskSpec.starter(h=7, freq="D")

# Production — requires TSFM adapters, full 5-window backtest
spec = TaskSpec.production(h=14, freq="D")
```

## Diagnose Without Fitting

```python
from tsagentkit import diagnose

report = diagnose(df)
# Returns: {validation, qa_report, plan, route_decision, task_spec_used}
```

`diagnose()` runs validation, QA, and planning without fitting any models —
useful for checking data quality and seeing what the router would do.

## Repair Errors Programmatically

```python
from tsagentkit import repair, validate_contract
from tsagentkit.contracts.errors import EDSNotMonotonic

try:
    report = validate_contract(df)
    report.raise_if_errors()
except EDSNotMonotonic:
    df, actions = repair(df)
```

`repair()` re-validates and applies deterministic safe fixes automatically
(e.g., sorting, deduplicating, dropping future null rows).

## Environment Check

```bash
# Check installed dependencies and adapter status
python -m tsagentkit doctor

# Print version
python -m tsagentkit version

# Machine-readable API schema
python -m tsagentkit describe
```

## Next Steps

- **Recipes**: `skill/recipes.md` — end-to-end runnable templates
- **API Reference**: `skill/tool_map.md` — task-to-API lookup table
- **Troubleshooting**: `skill/TROUBLESHOOTING.md` — error codes and fix hints
- **Architecture**: `docs/ARCHITECTURE.md` — system design and layering
- **API Stability**: `docs/API_STABILITY.md` — compatibility guarantees
