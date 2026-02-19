# tsagentkit Recipes

## What
Runnable templates for common production forecasting tasks.

## When
Use these recipes when an agent needs a correct baseline flow for time-series forecasting.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `config`: `ForecastConfig`
- Optional: covariates via `CovariateSet`

---

## Recipe 1: Retail Daily Sales (Assembly-First)
```python
import pandas as pd
from tsagentkit import (
    ForecastConfig,
    validate,
    build_dataset,
    make_plan,
    fit_all,
    predict_all,
    ensemble,
)

# Configuration
config = ForecastConfig(h=14, freq="D", quantiles=(0.1, 0.5, 0.9))

# Validate and prepare data
df = validate(raw_df)
dataset = build_dataset(df, config)

# Get TSFM models
models = make_plan(tsfm_only=True)

# Fit models (with caching)
artifacts = fit_all(models, dataset, device=config.device)

# Generate predictions
predictions = predict_all(models, artifacts, dataset, h=config.h, quantiles=config.quantiles)

# Ensemble
result = ensemble(
    predictions,
    method=config.ensemble_method,
    quantiles=config.quantiles,
)

print(result.df)
```

## Recipe 2: Quick Forecast Wrapper
```python
from tsagentkit import forecast

# Zero-config forecast
result = forecast(df, h=7, freq="D")
print(result.df)

# With quantiles
result = forecast(df, h=14, freq="D", quantiles=(0.25, 0.5, 0.75))
print(result.df)
```

## Recipe 3: Batch Processing with Model Cache
```python
from tsagentkit import forecast, ModelCache
from tsagentkit.models.registry import REGISTRY, list_models

# Preload all TSFMs once
tsfm_models = [REGISTRY[name] for name in list_models(tsfm_only=True)]
ModelCache.preload(tsfm_models, device="cuda")

# Process many datasets efficiently
results = []
for batch_df in batches:
    result = forecast(batch_df, h=7)  # Uses cached models
    results.append(result)

# Cleanup when done
ModelCache.unload()
```

## Recipe 4: Custom Covariates
```python
from tsagentkit import ForecastConfig, validate, build_dataset, run_forecast
from tsagentkit.core.dataset import CovariateSet

# Prepare covariates
static_cov = pd.DataFrame({
    "unique_id": ["A", "B"],
    "category": ["retail", "online"],
})

future_cov = pd.DataFrame({
    "unique_id": ["A", "A", "B", "B"],
    "ds": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-08", "2024-01-09"]),
    "promotion": [1, 0, 1, 1],
})

covariates = CovariateSet(static=static_cov, future=future_cov)

# Run forecast with covariates
config = ForecastConfig(h=2, freq="D")
df = validate(raw_df)
dataset = build_dataset(df, config, covariates)
result = run_forecast(df, config, covariates)
```

## Recipe 5: Robust Error Handling
```python
from tsagentkit import forecast, ForecastConfig
from tsagentkit.core.errors import EContract, ENoTSFM, EInsufficient, TSAgentKitError

config = ForecastConfig(h=7, freq="D")

try:
    result = forecast(df, h=7)
except EContract as e:
    # Fix data format issues
    print(f"Contract error: {e.message}")
    print(f"Hint: {e.fix_hint}")
except ENoTSFM:
    # No TSFMs available - check installation
    print("No TSFMs available. Install: pip install tsagentkit[chronos]")
except EInsufficient as e:
    # Not enough models succeeded
    print(f"Insufficient models: {e.message}")
except TSAgentKitError as e:
    # Other errors
    print(f"Error [{e.code}]: {e.message}")
```

## Recipe 6: Inspect Model Registry
```python
from tsagentkit import list_models, check_health
from tsagentkit.models.registry import REGISTRY, ModelSpec

# List available models
print("TSFMs:", list_models(tsfm_only=True))
print("All models:", list_models())

# Check health
health = check_health()
print(f"Available: {health.tsfm_available}")
print(f"All OK: {health.all_ok}")

# Get model specs
chronos_spec = REGISTRY["chronos"]
print(f"Chronos context limit: {chronos_spec.max_context_length}")
```

## Recipe 7: Length Limit Management
```python
from tsagentkit import ForecastConfig
from tsagentkit.models.length_utils import (
    check_data_compatibility,
    get_effective_limits,
    adjust_context_length,
)
from tsagentkit.models.registry import REGISTRY

# Check data compatibility with models
config = ForecastConfig(h=7, freq="D")
chronos_spec = REGISTRY["chronos"]

# Get effective limits
limits = get_effective_limits(chronos_spec, config)
print(f"Max context: {limits.max_context}")
print(f"Max prediction: {limits.max_prediction}")

# Check compatibility
compatible, msg = check_data_compatibility(chronos_spec, series_length=5000, h=7)
if not compatible:
    print(f"Compatibility issue: {msg}")
```

## Notes
- Data is automatically sorted by (`unique_id`, `ds`)
- TSFM models are cached automatically; use `ModelCache` for explicit control
- All forecasts return probabilistic quantiles by default
