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

## Assembly-First Pipeline

```python
from tsagentkit import (
    ForecastConfig,
    build_dataset,
    fit_all,
    make_plan,
    predict_all,
    validate,
)
from tsagentkit.models.ensemble import ensemble_with_quantiles

config = ForecastConfig(h=7, freq="D")
df = validate(raw_df)
dataset = build_dataset(df, config)
models = make_plan(tsfm_only=True)
artifacts = fit_all(models, dataset)
predictions = predict_all(models, artifacts, dataset, h=config.h)
ensemble_df = ensemble_with_quantiles(predictions, method=config.ensemble_method, quantiles=config.quantiles)
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
- `ForecastConfig`, `ForecastResult`, `TSDataset`, `CovariateSet`
- `validate`, `build_dataset`, `make_plan`, `fit_all`, `predict_all`, `ensemble`
- `ModelCache`
- `REGISTRY`, `ModelSpec`, `list_models`
- `check_health`
- `TSAgentKitError`, `EContract`, `ENoTSFM`, `EInsufficient`, `ETemporal`

Model protocol API (`from tsagentkit.models import ...`):
- `fit`, `predict`
- `ensemble`, `ensemble_with_quantiles`
- `get_spec`, `list_available`, `list_models`

Inspection API (`from tsagentkit.inspect import ...`):
- `list_models`
- `check_health`

## Errors

Core error types:
- `EContract`: input contract violations
- `ENoTSFM`: no TSFM adapters available
- `EInsufficient`: not enough successful model outputs
- `ETemporal`: temporal integrity violations

```python
from tsagentkit import TSAgentKitError, forecast

try:
    result = forecast(raw_df, h=7)
except TSAgentKitError as e:
    print(e.error_code, e.fix_hint)
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
