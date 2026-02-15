# tsagentkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Deterministic execution engine for time-series forecasting, designed to be called by coding agents.

tsagentkit gives you:
- a strict panel data contract (`unique_id`, `ds`, `y`)
- always-on temporal guardrails (no leakage, no random splits)
- deterministic routing/fallback and reproducible run artifacts
- both quick wrappers and assembly-first low-level APIs

## Start Here

### Install

```bash
pip install tsagentkit
```

### Run a Forecast

```python
from tsagentkit import forecast

# df must contain columns: unique_id, ds, y
result = forecast(df, horizon=7, freq="D")
print(result.forecast.df.head())
```

### Verify Environment

```bash
python -m tsagentkit doctor
python -m tsagentkit version
python -m tsagentkit describe
```

## Choose Your API Level

| If you want... | Use | Typical code size |
|---|---|---|
| Fastest path from dataframe to forecast | `forecast(df, horizon=...)` | 1-2 lines |
| Full run artifact with standard orchestration | `run_forecast(df, spec, mode=...)` | 2-4 lines |
| Full control and inspectable intermediate artifacts | assembly-first pipeline (`validate` -> `qa` -> `plan` -> `fit/predict`) | 10+ lines |

## Data Contract

Required input columns:
- `unique_id`: series identifier
- `ds`: timestamp
- `y`: numeric target

Example shape:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "unique_id": ["A", "A", "A", "B", "B", "B"],
        "ds": pd.date_range("2025-01-01", periods=3, freq="D").tolist() * 2,
        "y": [10, 12, 11, 30, 29, 31],
    }
)
```

## Core Guardrails

These are enforced by design:
- `E_SPLIT_RANDOM_FORBIDDEN`: no random train/test splits on time series
- `E_DS_NOT_MONOTONIC`: timestamps must be monotonic within each series
- `E_COVARIATE_LEAKAGE`: observed covariates cannot leak into the future
- `E_TSFM_REQUIRED_UNAVAILABLE`: TSFM policy required adapters but none are available

Basic error handling pattern:

```python
from tsagentkit import TSAgentKitError, forecast

try:
    result = forecast(df, horizon=7)
except TSAgentKitError as e:
    print(e.error_code, e.fix_hint)
    raise
```

## Workflows

### Assembly-First (Recommended)

```python
from tsagentkit import (
    TaskSpec,
    build_dataset,
    fit,
    make_plan,
    package_run,
    predict,
    run_qa,
    validate_contract,
)

spec = TaskSpec.production(h=14, freq="D")

validation = validate_contract(df)
validation.raise_if_errors()

qa_report = run_qa(df, spec, mode="standard")
dataset = build_dataset(df, spec)
plan, _route_decision = make_plan(dataset, spec)
model_artifact = fit(dataset, plan)
forecast_result = predict(dataset, model_artifact, spec)

run_artifact = package_run(
    forecast=forecast_result,
    plan=plan,
    task_spec=spec.model_dump(),
    qa_report=qa_report,
    model_artifact=model_artifact,
    provenance=forecast_result.provenance,
)
```

### Convenience Wrapper (`run_forecast`)

```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec.production(h=14, freq="D")
result = run_forecast(df, spec, mode="standard")
print(result.forecast.df.head())
```

### Diagnose and Auto-Repair

```python
from tsagentkit import diagnose, repair

report = diagnose(df, horizon=7)
print(report["validation"]["valid"])

repaired_df, actions = repair(df)
print(actions)
```

### Hierarchical Forecasting

Hierarchy APIs live in `tsagentkit.hierarchy`:

```python
import pandas as pd

from tsagentkit import TaskSpec, run_forecast
from tsagentkit.hierarchy import HierarchyStructure

df_with_hierarchy = df.assign(region="all", store=df["unique_id"])
hierarchy = HierarchyStructure.from_dataframe(
    df=df_with_hierarchy,
    hierarchy_columns=["region", "store"],
)

spec = TaskSpec.production(h=7, freq="D")
result = run_forecast(
    df,
    spec,
    hierarchy=hierarchy,
    reconciliation_method="min_trace",
)
```

### Artifact Lifecycle

```python
from tsagentkit import (
    load_run_artifact,
    replay_forecast_from_artifact,
    save_run_artifact,
    validate_run_artifact_for_serving,
)

save_run_artifact(run_artifact, "run.json")
loaded = load_run_artifact("run.json")
validate_run_artifact_for_serving(
    loaded,
    expected_task_signature=loaded.provenance.task_signature,
)
replayed = replay_forecast_from_artifact(loaded)
```

## TSFM Policy

```python
from tsagentkit import TaskSpec

# default in TaskSpec(): required
spec_required = TaskSpec(h=7, freq="D")

# allow fallback to non-TSFM baselines
spec_preferred = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})

# disable TSFM routing entirely
spec_disabled = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "disabled"})
```

## API Reference (Imports)

```python
from tsagentkit import (
    # High-level
    forecast,
    diagnose,
    repair,
    run_forecast,
    TaskSpec,
    describe,
    # Assembly-first
    validate_contract,
    run_qa,
    build_dataset,
    make_plan,
    build_plan_graph,
    attach_plan_graph,
    fit,
    predict,
    package_run,
    # Lifecycle
    save_run_artifact,
    load_run_artifact,
    validate_run_artifact_for_serving,
    replay_forecast_from_artifact,
    # Adapter capability
    get_adapter_capability,
    list_adapter_capabilities,
)

from tsagentkit.hierarchy import (
    HierarchyStructure,
    reconcile_forecasts,
)
```

## Developer Commands

```bash
uv sync
uv run python -m pytest
TSFM_RUN_REAL=1 uv run python -m pytest -m tsfm
```

## Documentation Map

- `skill/QUICKSTART.md`: 3-minute guide for agent users
- `skill/README.md`: agent-facing workflow and module map
- `skill/recipes.md`: end-to-end recipes
- `skill/tool_map.md`: task-to-API lookup
- `skill/TROUBLESHOOTING.md`: error codes and fixes
- `docs/ARCHITECTURE.md`: technical architecture
- `docs/PRD.md`: full product requirements

## License

Apache-2.0
