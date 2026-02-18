# tsagentkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Minimalist time-series forecasting toolkit for coding agents.

`tsagentkit` provides:
- a strict, fixed panel data contract
- zero-config TSFM ensemble forecasting
- a small set of pipeline primitives for agent customization
- explicit TSFM model lifecycle control via `ModelCache`

## Install

```bash
pip install tsagentkit
```

## Fixed Data Contract

Input data must use exactly these columns:
- `unique_id`: series identifier
- `ds`: timestamp
- `y`: target value

Custom column remapping is not supported.

```python
import pandas as pd

# Valid input schema
raw_df = pd.DataFrame({
    "unique_id": ["A"] * 30,
    "ds": pd.date_range("2025-01-01", periods=30, freq="D"),
    "y": range(30),
})
```

## Quick Start

```python
from tsagentkit import forecast

result = forecast(raw_df, h=7, freq="D")
print(result.df.head())
```

## Standard Pipeline API

```python
from tsagentkit import ForecastConfig, run_forecast

config = ForecastConfig(h=7, freq="D", ensemble_method="median")
result = run_forecast(raw_df, config)
print(result.df.head())
```

## Building-Block Pipeline

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
print(result.df.head())
```

## Model Cache Lifecycle

`ModelCache` is the single source of truth for loaded TSFM instances.

```python
from tsagentkit import ModelCache, forecast
from tsagentkit.models.registry import REGISTRY

# Optional preload
models = [m for m in REGISTRY.values() if m.is_tsfm]
ModelCache.preload(models)

# Reuses cached models across calls
result = forecast(raw_df, h=7)

# Explicit release
ModelCache.unload()           # all models
# ModelCache.unload("chronos")  # one model
```

`ModelCache.unload()` semantics:
- releases all `tsagentkit`-owned model references
- calls adapter unload hooks when available
- triggers best-effort backend cleanup (`gc.collect`, CUDA/MPS cache clear)
- cannot reclaim memory still referenced by external user code

## Public API

Top-level (`from tsagentkit import ...`):
- `forecast`, `run_forecast`
- `ForecastConfig`, `ForecastResult`, `RunResult`
- `TSDataset`, `CovariateSet`
- `validate`, `build_dataset`, `make_plan`, `fit_all`, `predict_all`, `ensemble`
- `ModelCache`
- `REGISTRY`, `ModelSpec`, `list_models`
- `resolve_device`
- `check_health`
- `TSAgentKitError`, `EContract`, `ENoTSFM`, `EInsufficient`, `ETemporal`

Inspection API (`from tsagentkit.inspect import ...`):
- `list_models`
- `check_health`, `HealthReport`

## Errors

Core error types:
- `EContract`: input contract violations
- `ENoTSFM`: TSFM registry invariant violation (internal misconfiguration)
- `EInsufficient`: not enough successful model outputs
- `ETemporal`: temporal integrity violations

```python
from tsagentkit import EContract, forecast

try:
    result = forecast(raw_df, h=7)
except EContract as e:
    print(e.code, e.hint)
    raise
```

## Developer Commands

```bash
uv sync --all-extras
uv run pytest
uv run mypy src/tsagentkit
uv run ruff format src/
```

## License

Apache-2.0
