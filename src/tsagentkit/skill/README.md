# tsagentkit Skill Documentation

> **For AI Agents**: Quick reference guide for using tsagentkit correctly.

## Overview

tsagentkit is a Python library that enforces proper time-series forecasting practices through a strict workflow pipeline. It prevents common mistakes like data leakage and random train/test splits.

## Workflow Pipeline

```
validate -> QA -> series -> route -> backtest -> fit -> predict -> package
```

Use `run_forecast()` for the complete pipeline, or individual modules for custom workflows.

## Quick Start

```python
import pandas as pd
from tsagentkit import TaskSpec, run_forecast

# Prepare data (must have unique_id, ds, y columns)
df = pd.DataFrame({
    "unique_id": ["A", "A", "B", "B"],
    "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
    "y": [1.0, 2.0, 3.0, 4.0],
})

# Define task
spec = TaskSpec(horizon=7, freq="D")

# Run forecast
result = run_forecast(df, spec, mode="standard")

# Access results
print(result.forecast)
print(result.summary())
```

## Guardrails (CRITICAL)

tsagentkit enforces these guardrails - violations will raise errors:

| Guardrail | Error Code | Prevention |
|-----------|------------|------------|
| No random splits | `E_SPLIT_RANDOM_FORBIDDEN` | Data must be sorted by (unique_id, ds) |
| No covariate leakage | `E_COVARIATE_LEAKAGE` | Future covariates cannot be used |
| Temporal ordering | `E_CONTRACT_UNSORTED` | Dates must increase within each series |

## Module Reference

### contracts - Data Validation

**What**: Validates input data schema and types.

**When to use**: Before any forecasting operation.

**Key Functions**:
```python
from tsagentkit import validate_contract, TaskSpec

# Validate data
report = validate_contract(df)
if not report.valid:
    report.raise_if_errors()

# Create task spec
spec = TaskSpec(horizon=7, freq="D", quantiles=[0.1, 0.5, 0.9])
```

**Input Requirements**:
- `unique_id` (str): Series identifier
- `ds` (datetime): Timestamp
- `y` (numeric): Target value

### series - Time Series Handling

**What**: Immutable dataset container with sparsity profiling.

**When to use**: After validation, before modeling.

**Key Classes/Functions**:
```python
from tsagentkit import TSDataset, build_dataset

# Build dataset
dataset = build_dataset(df, spec)

# Access properties
print(dataset.n_series)      # Number of series
print(dataset.date_range)    # (start, end) dates

# Split train/test
train, test = dataset.split_train_test(test_size=7)

# Filter by series
filtered = dataset.filter_series(["A", "B"])

# Filter by dates
filtered = dataset.filter_dates(start="2024-01-01", end="2024-01-31")
```

**Sparsity Classes**:
- `regular`: Normal series
- `intermittent`: Many zero values (demand forecasting)
- `sparse`: Irregular gaps
- `cold_start`: Very few observations

### router - Model Selection

**What**: Selects appropriate models based on data characteristics.

**When to use**: After building dataset, before training.

**Key Functions**:
```python
from tsagentkit import make_plan, Plan, FallbackLadder

# Auto-select plan
plan = make_plan(dataset, spec)

# Access plan
print(plan.primary_model)      # e.g., "SeasonalNaive"
print(plan.fallback_chain)     # e.g., ["HistoricAverage", "Naive"]
print(plan.signature)          # Hash for provenance

# Manual fallback ladder
ladder = FallbackLadder.get_ladder(
    is_intermittent=False,
    is_cold_start=False
)
```

**Routing Strategies**:
- `auto`: Automatic selection based on data
- `baseline_only`: Use only baseline models
- `tsfm_first`: Try foundation models first (v0.2+)

### backtest - Rolling Window Validation

**What**: Temporal cross-validation without random splits.

**When to use**: In standard/strict mode before final training.

**Key Functions**:
```python
from tsagentkit import rolling_backtest, BacktestReport

# Define fit/predict functions
def fit_func(model_name, data, config):
    # Return fitted model
    return model

def predict_func(model, data, horizon):
    # Return predictions DataFrame
    return predictions

# Run backtest
report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit_func,
    predict_func=predict_func,
    n_windows=5,
    window_strategy="expanding",  # or "sliding"
)

# Access results
print(report.aggregate_metrics)
print(report.get_best_series("wape"))
print(report.summary())
```

**Available Metrics**:
- `wape`: Weighted Absolute Percentage Error
- `smape`: Symmetric Mean Absolute Percentage Error
- `mase`: Mean Absolute Scaled Error
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Squared Error
- `pinball_X.XX`: Pinball loss for quantiles

### serving - Main Orchestration

**What**: Complete pipeline execution with provenance.

**When to use**: For end-to-end forecasting.

**Key Functions**:
```python
from tsagentkit import run_forecast

# Quick mode (no backtest)
result = run_forecast(df, spec, mode="quick")

# Standard mode (with backtest)
result = run_forecast(
    df, spec,
    mode="standard",
    fit_func=custom_fit,
    predict_func=custom_predict,
)

# Access results
print(result.forecast)           # Predictions DataFrame
print(result.model_name)         # Model that succeeded
print(result.provenance)         # Full lineage
print(result.summary())          # Human-readable summary
```

**Execution Modes**:
- `quick`: Skip backtest, fastest
- `standard`: Full pipeline with backtest (recommended)
- `strict`: Fail on any QA issue

## Error Handling

All errors inherit from `TSAgentKitError` and include:
- `error_code`: String code for programmatic handling
- `message`: Human-readable description
- `context`: Additional debugging information

```python
from tsagentkit import TSAgentKitError

try:
    result = run_forecast(df, spec)
except TSAgentKitError as e:
    print(f"Error {e.error_code}: {e.message}")
    print(f"Context: {e.context}")
```

## Common Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| `E_CONTRACT_MISSING_COLUMN` | Required column missing | Add unique_id, ds, y columns |
| `E_CONTRACT_INVALID_TYPE` | Wrong column type | Convert types (e.g., to_datetime) |
| `E_CONTRACT_DUPLICATE_KEY` | Duplicate (unique_id, ds) pairs | Remove duplicates |
| `E_SPLIT_RANDOM_FORBIDDEN` | Data not sorted | Sort: `df.sort_values(['unique_id', 'ds'])` |
| `E_COVARIATE_LEAKAGE` | Future covariates detected | Remove future-dated covariates |
| `E_FALLBACK_EXHAUSTED` | All models failed | Check data quality, try different models |

## Best Practices

1. **Always validate first**: Call `validate_contract()` before custom workflows
2. **Use TSDataset**: Wrap data for immutability guarantees
3. **Leverage sparsity profile**: Router uses this for model selection
4. **Check provenance**: Always review `result.provenance` for reproducibility
5. **Handle fallbacks**: Monitor `fallbacks_triggered` in provenance
6. **Use appropriate mode**:
   - Development: `standard` (with backtest)
   - Production API: `quick` (fastest)
   - Critical applications: `strict` (max safety)

## File Locations

- Source: `src/tsagentkit/`
- Tests: `tests/`
- Recipes: `src/tsagentkit/skill/recipes.md`

## See Also

- `recipes.md`: Complete runnable examples
- `docs/PRD.md`: Technical requirements
- `docs/DEV_PLAN_v0.1.md`: Implementation roadmap
