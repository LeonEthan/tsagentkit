# tsagentkit Tool Map

## What
Task-to-API lookup for coding agents building production forecasting systems.

## When
Use this map to choose the minimal stable API surface for each step.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `config`: `ForecastConfig`
- Optional: covariates via `CovariateSet`

## Workflow
1. Validate and build dataset
2. Get models from registry
3. Fit and predict
4. Ensemble results

---

## Stable APIs

### Main Entry Points
- `forecast(data, h, freq, **kwargs)` - Zero-config forecast
- `run_forecast(data, config, covariates)` - Config-based forecast

### Pipeline Building Blocks
- `validate(df, config)` - Validate input data
- `build_dataset(df, config, covariates)` - Build TSDataset
- `make_plan(tsfm_only=True)` - Get model list from registry
- `fit_all(models, dataset, device)` - Fit all models
- `predict_all(models, artifacts, dataset, h, quantiles)` - Generate predictions
- `ensemble(predictions, method, quantiles)` - Aggregate ensemble

### Core Types
- `ForecastConfig` - Configuration (h, freq, quantiles, etc.)
- `TSDataset` - Time-series dataset container
- `CovariateSet` - Static/past/future covariates
- `ForecastResult` - Forecast output with `.df`
- `ModelSpec` - Model specification from registry

### Registry
- `REGISTRY` - Dictionary of all models
- `list_models(tsfm_only=False)` - List available models

### Model Cache
- `ModelCache.get(spec, device)` - Get cached model
- `ModelCache.preload(models, device)` - Preload models
- `ModelCache.unload(model_name=None)` - Unload models
- `ModelCache.list_loaded()` - List cached models

### Inspection
- `check_health()` - Health check report
- `list_models(tsfm_only=True)` - List models

### Length Utilities
- `check_data_compatibility(spec, series_length, h)` - Check compatibility
- `get_effective_limits(spec, config)` - Get context/prediction limits
- `adjust_context_length(df, max_length)` - Adjust data length
- `validate_prediction_length(h, spec, config)` - Validate horizon

---

## Task -> API Mapping

| Task | Primary API | Notes |
|---|---|---|
| Quick forecast | `forecast(df, h=7)` | Zero-config, returns ForecastResult |
| Config-based forecast | `run_forecast(df, config)` | Full config control |
| Data validation | `validate(df)` | Checks columns, nulls |
| Build dataset | `build_dataset(df, config)` | Creates TSDataset |
| Get models | `make_plan(tsfm_only=True)` | Returns list of ModelSpec |
| Fit models | `fit_all(models, dataset)` | Returns artifacts list |
| Generate predictions | `predict_all(models, artifacts, dataset, h)` | Returns list of DataFrames |
| Ensemble | `ensemble(predictions, method, quantiles)` | Aggregates predictions |
| Preload models | `ModelCache.preload(models)` | For batch processing |
| Check health | `check_health()` | Returns HealthReport |
| List models | `list_models(tsfm_only=True)` | TSFM or all models |

---

## Error Handling

| Error | Code | When |
|---|---|---|
| `EContract` | `E_CONTRACT` | Wrong columns, nulls, empty data |
| `ENoTSFM` | `E_NO_TSFM` | No TSFMs registered |
| `EInsufficient` | `E_INSUFFICIENT` | Too few models succeeded |
| `ETemporal` | `E_TEMPORAL` | Sorting/covariate issues |

---

## Related Docs
- `skill/README.md` - Overview and patterns
- `skill/QUICKSTART.md` - Getting started
- `skill/recipes.md` - End-to-end templates
- `skill/TROUBLESHOOTING.md` - Error reference
- `docs/DESIGN.md` - Architecture details
