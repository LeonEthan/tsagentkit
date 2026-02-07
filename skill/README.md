# tsagentkit Skill Documentation

> **For AI Agents**: Guide to using tsagentkit for time-series forecasting tasks.

## Overview

`tsagentkit` is a robust execution engine for time-series forecasting. It enforces proper time-series practices and provides guardrails to prevent common mistakes like data leakage and random train/test splits.

## Quick Start

```python
import pandas as pd
from tsagentkit import TaskSpec, run_forecast

# Prepare your data
df = pd.DataFrame({
    "unique_id": ["A", "A", "A", "B", "B", "B"],
    "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
    "y": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
})

# Define the task
spec = TaskSpec(h=7, freq="D")

# Run forecast
result = run_forecast(df, spec)
print(result.forecast.df.head())
```

## Core Workflow

```
validate -> QA -> series -> route -> backtest -> fit -> predict -> package
```

## Module Guide

### 1. Contracts (`tsagentkit.contracts`)

**What**: Data validation and task specifications.

**When to use**: Always start here to validate input data.

**Key Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `validate_contract(data)` | Validate data schema | DataFrame with unique_id, ds, y | ValidationReport |
| `TaskSpec(...)` | Define forecasting task | h, freq, quantiles, etc. | TaskSpec |

**Example**:
```python
from tsagentkit import validate_contract, TaskSpec

# Validate first
report = validate_contract(df)
if not report.valid:
    report.raise_if_errors()

# Then create task spec
spec = TaskSpec(
    h=14,
    freq="D",
    quantiles=[0.1, 0.5, 0.9],
)
```

### 2. Series (`tsagentkit.series`)

**What**: Time series data structures and operations.

**When to use**: When you need to work with time series datasets directly.

**Key Classes/Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `TSDataset.from_dataframe(df, spec)` | Create dataset | DataFrame, TaskSpec | TSDataset |
| `build_dataset(df, spec)` | Convenience wrapper | DataFrame, TaskSpec | TSDataset |

**Example**:
```python
from tsagentkit.series import TSDataset

dataset = TSDataset.from_dataframe(df, spec)
print(f"Series: {dataset.n_series}, Observations: {dataset.n_observations}")
```

### 3. QA (`tsagentkit.qa`)

**What**: Data quality checks and leakage detection.

**When to use**: Before modeling to check for issues.

**Key Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `run_qa(data, spec, mode)` | Quality checks | DataFrame, TaskSpec, mode | QAReport |

**Modes**:
- `"quick"`: Basic checks
- `"standard"`: Full checks with auto-repair
- `"strict"`: Fail on any issue

**Example**:
```python
from tsagentkit.qa import run_qa

qa_report = run_qa(df, spec, mode="standard")
if qa_report.leakage_detected:
    print("Warning: Potential data leakage detected")
```

### 4. Router (`tsagentkit.router`)

**What**: Model selection and fallback strategies.

**When to use**: To get model recommendations or create execution plans.

**Key Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `make_plan(dataset, spec)` | Create execution plan | TSDataset, TaskSpec | `(PlanSpec, RouteDecision)` |
| `execute_with_fallback(fit_func, dataset, plan)` | Execute with fallback | fit function, dataset, plan | (result, model_name) |

**Example**:
```python
from tsagentkit.router import make_plan

plan, route_decision = make_plan(dataset, spec)
print(f"Candidates: {plan.candidate_models}")
print(f"Buckets: {route_decision.buckets}")
```

### 5. Models (`tsagentkit.models`)

**What**: Model fitting and prediction.

**When to use**: To fit models directly (usually handled by `run_forecast`).

**Key Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `fit(dataset, plan)` | Fit model | TSDataset, PlanSpec | ModelArtifact |
| `predict(dataset, artifact, spec)` | Generate forecast | TSDataset, ModelArtifact, TaskSpec | ForecastResult |

**Example**:
```python
from tsagentkit.models import fit, predict

artifact = fit(dataset, plan)
forecast = predict(dataset, artifact, spec)
```

### 6. Backtest (`tsagentkit.backtest`)

**What**: Rolling window backtesting with temporal integrity.

**When to use**: To evaluate model performance without data leakage.

**Key Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `rolling_backtest(dataset, spec, plan)` | Temporal cross-validation | TSDataset, TaskSpec, PlanSpec | BacktestReport |

**Example**:
```python
from tsagentkit.backtest import rolling_backtest

report = rolling_backtest(dataset, spec, plan, n_windows=5)
print(f"WAPE: {report.aggregate_metrics['wape']:.2%}")
```

### 7. Serving (`tsagentkit.serving`)

**What**: Complete forecasting pipeline orchestration.

**When to use**: Use `run_forecast()` for the full workflow.

**Key Functions**:

| Function | Purpose | Inputs | Output |
|----------|---------|--------|--------|
| `run_forecast(data, spec, mode)` | Complete pipeline | DataFrame, TaskSpec, mode | RunArtifact |
| `package_run(...)` | Package all outputs | forecast, plan, backtest, etc. | RunArtifact |

**Example**:
```python
from tsagentkit import run_forecast

result = run_forecast(df, spec, mode="standard")
print(result.summary())
```

## Guardrails (Error Prevention)

| Error Code | Meaning | How to Fix |
|------------|---------|------------|
| `E_CONTRACT_MISSING_COLUMN` | Missing required column | Add unique_id, ds, y columns |
| `E_CONTRACT_DUPLICATE_KEY` | Duplicate (unique_id, ds) pairs | Remove duplicates |
| `E_SPLIT_RANDOM_FORBIDDEN` | Data not sorted or random splits detected | Use `df.sort_values(['unique_id', 'ds'])` or use temporal splits only |
| `E_COVARIATE_LEAKAGE` | Future covariates leaked | Mark covariates as known/observed correctly |
| `E_MODEL_FIT_FAIL` | Model training failed | Check data quality or use fallback |
| `E_FALLBACK_EXHAUSTED` | All models failed | Check data is valid for forecasting |
| `E_DS_NOT_MONOTONIC` | Data is not sorted by time | Sort by `unique_id`, `ds` before forecasting |

## Common Patterns

### Pattern 1: Quick Forecast (No Backtest)
```python
result = run_forecast(df, spec, mode="quick")
forecast_df = result.forecast.df
```

### Pattern 2: Full Pipeline with Metrics
```python
result = run_forecast(df, spec, mode="standard")
print(f"Backtest WAPE: {result.backtest_report['aggregate_metrics']['wape']:.2%}")
```

### Pattern 3: Custom Model Integration
```python
def my_fit(dataset, plan):
    # Your custom fitting logic
    return ModelArtifact(model=my_model, model_name="Custom", config={})

def my_predict(dataset, artifact, spec):
    # Your custom prediction logic
    return ForecastResult(df=forecast_df, provenance=prov, model_name="Custom", horizon=spec.horizon)

result = run_forecast(df, spec, fit_func=my_fit, predict_func=my_predict)
```

### Pattern 4: Hierarchical Forecasting
```python
from tsagentkit.hierarchy import HierarchyStructure

# Define hierarchy
hierarchy = HierarchyStructure(
    aggregation_graph={"Total": ["A", "B"]},
    bottom_nodes=["A", "B"],
)

result = run_forecast(df, spec, hierarchy=hierarchy)
```

## Data Format Requirements

Your input DataFrame **must** have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | string | Series identifier (e.g., "store_1", "product_A") |
| `ds` | datetime | Timestamp of observation |
| `y` | numeric | Target value to forecast |

**Optional columns** (for covariates):
- Known in advance: Can use in future periods (e.g., holidays, promotions)
- Observed: Only known up to forecast time (e.g., temperature, stock price)

## Next Steps

- See `recipes.md` for complete runnable examples
- See `tool_map.md` for detailed function reference
- Check the PRD at `docs/PRD.md` for technical specifications
