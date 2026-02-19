# tsagentkit Quickstart

Get a forecast running in under 3 minutes.

---

## Installation

```bash
pip install tsagentkit
```

## Minimal Example (2 lines)

```python
from tsagentkit import forecast

result = forecast(df, h=7)
```

`forecast()` validates data, builds dataset, fits TSFMs, and returns ensemble forecast.
Input DataFrame must have columns: `unique_id`, `ds`, `y`.

## Standard Pipeline Example (10 lines)

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

config = ForecastConfig(h=7, freq="D")
df = validate(raw_df)
dataset = build_dataset(df, config)
models = make_plan(tsfm_only=True)
artifacts = fit_all(models, dataset)
predictions = predict_all(models, artifacts, dataset, h=config.h)
result = ensemble(predictions, method=config.ensemble_method, quantiles=config.quantiles)
```

## Config Presets

| Preset | Use Case |
|--------|----------|
| `ForecastConfig.quick(h, freq)` | Quick experimentation |
| `ForecastConfig.strict(h, freq)` | Fail if TSFM unavailable |

```python
from tsagentkit import ForecastConfig

# Quick experimentation
config = ForecastConfig.quick(h=7, freq="D")

# Production - requires TSFMs
config = ForecastConfig.strict(h=14, freq="H")
```

## Configuration Options

```python
from tsagentkit import ForecastConfig

config = ForecastConfig(
    h=7,                           # Forecast horizon
    freq="D",                      # Frequency: 'D', 'H', 'M', etc.
    quantiles=(0.1, 0.5, 0.9),     # Probabilistic forecast quantiles
    ensemble_method="median",      # "median" or "mean"
    min_tsfm=1,                    # Minimum TSFMs required
    device="auto",                 # "auto", "cuda", "mps", "cpu"
)
```

## Health Check

```python
from tsagentkit import check_health

health = check_health()
print(health)
# tsagentkit Health Report
# ========================================
# TSFMs available: chronos, timesfm, moirai, patchtst_fm
# Baselines available: True
# Overall: OK
```

## Next Steps

- **Recipes**: `skill/recipes.md` — end-to-end runnable templates
- **API Reference**: `skill/tool_map.md` — task-to-API lookup table
- **Troubleshooting**: `skill/TROUBLESHOOTING.md` — error codes and fix hints
- **Architecture**: `docs/DESIGN.md` — system design and internals
