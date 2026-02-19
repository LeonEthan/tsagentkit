# tsagentkit Skill Documentation

## What
Agent-facing guide for building production time-series forecasting systems with `tsagentkit`.
This skill is TSFM-first: coding agents should compose explicit steps for control,
while `forecast()` remains a convenience wrapper for quick results.

## When
Use this guide when an agent needs to:
- Validate panel data (unique_id, ds, y)
- Run TSFM ensemble forecasting
- Control model lifecycle and caching
- Build custom forecasting pipelines

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `config`: `ForecastConfig` (h, freq, quantiles, etc.)
- Optional: covariates via `CovariateSet`

## Workflow
1. `validate` - Check data format
2. `build_dataset` - Create `TSDataset`
3. `make_plan` - Get model list from registry
4. `fit_all` - Fit TSFMs (cached)
5. `predict_all` - Generate predictions
6. `ensemble` - Aggregate results

---

## Core Principles
- **TSFM-first**: Chronos, TimesFM, Moirai, PatchTST-FM are the primary models
- **Minimal API**: Simple config, clear pipeline steps
- **Model caching**: `ModelCache` avoids expensive TSFM reloads
- **Pure functions**: Functional composition over class hierarchies

## Guardrails
- `EContract`: Input data must have [unique_id, ds, y] columns
- `ENoTSFM`: TSFM registry invariant violation
- `EInsufficient`: Too few TSFMs succeeded for ensemble
- `ETemporal`: Temporal integrity violation (not sorted, future leakage)

## Module Map
- `core/config`: `ForecastConfig` - unified configuration
- `core/dataset`: `TSDataset`, `CovariateSet` - data containers
- `core/errors`: Error types (`EContract`, `ENoTSFM`, etc.)
- `models/registry`: `REGISTRY`, `ModelSpec`, `list_models`
- `models/cache`: `ModelCache` - TSFM lifecycle management
- `models/ensemble`: `ensemble` - prediction aggregation
- `pipeline`: Main pipeline functions

## Pattern 1: Zero-Config Forecast (Quickest)
```python
from tsagentkit import forecast

result = forecast(df, h=7, freq="D")
print(result.df)
```

## Pattern 2: Assembly-First Pipeline (Recommended)
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
from tsagentkit.models.registry import REGISTRY, list_models

# Configure
config = ForecastConfig(h=7, freq="D", quantiles=(0.1, 0.5, 0.9))

# Validate and build dataset
df = validate(raw_df)
dataset = build_dataset(df, config)

# Get models from registry
models = make_plan(tsfm_only=True)

# Fit and predict
artifacts = fit_all(models, dataset, device=config.device)
predictions = predict_all(models, artifacts, dataset, h=config.h, quantiles=config.quantiles)

# Ensemble
result = ensemble(
    predictions,
    method=config.ensemble_method,
    quantiles=config.quantiles,
)
```

## Pattern 3: Model Cache for Batch Processing
```python
from tsagentkit import forecast, ModelCache
from tsagentkit.models.registry import REGISTRY, list_models

# Preload models for batch processing
tsfm_models = [REGISTRY[name] for name in list_models(tsfm_only=True)]
ModelCache.preload(tsfm_models, device="cuda")

# Run forecasts (uses cached models)
for batch_df in batches:
    result = forecast(batch_df, h=7)

# Cleanup
ModelCache.unload()
```

## Pattern 4: Health Check and Inspection
```python
from tsagentkit import check_health, list_models

# Check available TSFMs
health = check_health()
print(health.tsfm_available)
print(health.tsfm_missing)

# List all models
print(list_models(tsfm_only=True))
```

## Data Requirements
- Required columns: `unique_id`, `ds`, `y`
- `ds` must be datetime-like
- Data is automatically sorted by (`unique_id`, `ds`)
- No null values allowed in required columns

## Error Handling
```python
from tsagentkit import forecast
from tsagentkit.core.errors import EContract, ENoTSFM, EInsufficient

try:
    result = forecast(df, h=7)
except EContract as e:
    print(f"Data format error: {e.message}")
    print(f"Fix hint: {e.fix_hint}")
except ENoTSFM as e:
    print("No TSFMs available")
except EInsufficient as e:
    print(f"Not enough models succeeded: {e.message}")
```

## Next Steps
- Use `skill/QUICKSTART.md` for getting started in 3 minutes
- Use `skill/recipes.md` for end-to-end templates
- Use `skill/tool_map.md` for task-to-API lookup
- Use `skill/TROUBLESHOOTING.md` for error codes and fix hints
- Use `docs/DESIGN.md` for detailed architecture
