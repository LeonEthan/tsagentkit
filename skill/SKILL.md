---
name: tsagentkit
description: |
  This skill provides guidance for building production time-series forecasting systems
  using tsagentkit. Use this skill when the user needs to:
  - Run time-series forecasting with TSFM (Time Series Foundation Models)
  - Validate panel data format (unique_id, ds, y columns)
  - Build custom forecasting pipelines with full control
  - Handle model caching and lifecycle management
  - Troubleshoot forecasting errors and data issues
  - Ensemble multiple TSFM predictions (Chronos, TimesFM, Moirai, PatchTST-FM)
---

# tsagentkit Skill

Production time-series forecasting toolkit for agentic workflows. TSFM-first ensemble approach with zero-config convenience and full pipeline control.

## When to Use

- User asks for time-series forecasting or prediction
- Working with panel data (multiple time series)
- Need probabilistic forecasts with quantiles
- Want to use foundation models (Chronos, TimesFM, Moirai, PatchTST-FM)
- Need to forecast sales, traffic, demand, or any temporal data

## Core Principles

1. **TSFM-First**: Time Series Foundation Models are primary; baselines are fallback
2. **Zero-Config**: `forecast(df, h=7)` just works
3. **Full Control**: Pipeline functions for custom workflows
4. **Model Caching**: Avoid expensive TSFM reloads via `ModelCache`
5. **Pure Functions**: Functional composition over class hierarchies

## Quick Start

```python
from tsagentkit import forecast

result = forecast(df, h=7, freq="D")
print(result.df)
```

Input DataFrame must have columns: `unique_id`, `ds`, `y`.

## Pipeline Assembly (Full Control)

```python
from tsagentkit import (
    ForecastConfig,
    validate,
    build_dataset,
    make_plan,
    fit_all,
    predict_all,
    ensemble,
)

# 1. Configure
config = ForecastConfig(h=7, freq="D", quantiles=(0.1, 0.5, 0.9))

# 2. Validate and build dataset
df = validate(raw_df)
dataset = build_dataset(df, config)

# 3. Get models from registry
models = make_plan(tsfm_only=True)

# 4. Fit models
artifacts = fit_all(models, dataset, device=config.device)

# 5. Generate predictions
predictions = predict_all(models, artifacts, dataset, h=config.h, quantiles=config.quantiles)

# 6. Ensemble results
result = ensemble(predictions, method=config.ensemble_method, quantiles=config.quantiles)
```

## Error Types to Handle

| Error | Meaning | Fix |
|-------|---------|-----|
| `EContract` | Input data format violation | Check columns [unique_id, ds, y] |
| `ENoTSFM` | No TSFMs available | Install TSFM dependencies |
| `EInsufficient` | Too few models succeeded | Check data quality or lower min_tsfm |
| `ETemporal` | Temporal integrity issue | Sort data, check for future leakage |

## Key Imports

```python
# Standard pipeline
from tsagentkit import forecast, run_forecast, ForecastConfig, ForecastResult

# Building blocks
from tsagentkit import validate, build_dataset, make_plan, fit_all, predict_all, ensemble

# Model lifecycle
from tsagentkit import ModelCache

# Inspection
from tsagentkit import list_models, check_health
```

## Config Presets

```python
# Quick experimentation
config = ForecastConfig.quick(h=7, freq="D")

# Production - requires TSFMs
config = ForecastConfig.strict(h=14, freq="H")
```

## Model Cache for Batch Processing

```python
from tsagentkit import forecast, ModelCache
from tsagentkit.models.registry import REGISTRY

# Preload once
tsfm_models = [m for m in REGISTRY.values() if m.is_tsfm]
ModelCache.preload(tsfm_models, device="cuda")

# Run many forecasts
for batch_df in batches:
    result = forecast(batch_df, h=7)

# Cleanup
ModelCache.unload()
```

## Reference Documentation

- `references/quickstart.md` - 3-minute getting started guide
- `references/recipes.md` - End-to-end runnable templates
- `references/tool_map.md` - Task-to-API lookup table
- `references/troubleshooting.md` - Error codes and fix hints

## Data Requirements

Required columns: `unique_id`, `ds`, `y`
- `unique_id`: Series identifier (string or int)
- `ds`: Datetime column
- `y`: Target values (numeric)
- Data is automatically sorted by (`unique_id`, `ds`)
