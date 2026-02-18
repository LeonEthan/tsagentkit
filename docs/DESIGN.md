# DESIGN.md

> Detailed design reference for `tsagentkit`.
> Migrated from the original long-form `AGENTS.md` to keep agent context usage low.

> **Design Philosophy**: Minimalist, research-ready time-series forecasting toolkit for vibe coding agents.
> **Core Tenet**: Less code, more clarity. ~2,000 lines for full TSFM ensemble capability.

## Project Overview

`tsagentkit` is an ultra-lightweight execution engine for time-series forecasting. It provides **zero-config TSFM ensemble** for production use, while exposing **granular pipeline functions** for agent customization.

**Version**: 2.0.0
**Core Lines**: ~2,000 (excluding adapters)
**Design Goal**: Research-ready code that's easy to understand, modify, and extend.

---

## Architecture (Nanobot-Inspired)

### Core Principles

1. **Radical Simplicity**: Every module has a single, clear responsibility
2. **Flat Structure**: Minimal nesting, clear imports
3. **Pure Functions**: Functional composition over class hierarchies
4. **Single Source of Truth**: One registry, one config, one way to do things
5. **Research-Ready**: Clean, readable code for easy modification

### Two Usage Modes

```
┌─────────────────────────────────────────────────────────────┐
│                    tsagentkit v2.0                          │
│                    ~2,000 lines core                        │
├─────────────────────────────┬───────────────────────────────┤
│  Standard Pipeline          │  Agent Building               │
│  (Zero Config)              │  (Full Control)               │
├─────────────────────────────┼───────────────────────────────┤
│                             │                               │
│  forecast(df, h=7)          │  validate(df)                 │
│  ↓                          │  build_dataset(df, cfg)       │
│  TSFM Ensemble Only         │  plan = make_plan(tsfm_only=True) │
│  ↓                          │  artifacts = fit_all(plan)    │
│  Result                     │  preds = predict_all(...)     │
│                             │  ensemble(preds)              │
│                             │                               │
└─────────────────────────────┴───────────────────────────────┘
```

### Module Structure (Flat & Clean)

```
src/tsagentkit/
├── core/
│   ├── config.py          # Single ForecastConfig (frozen dataclass)
│   ├── dataset.py         # TSDataset (lightweight wrapper)
│   ├── types.py           # Shared type definitions
│   └── errors.py          # 4 error types only
│
├── models/
│   ├── registry.py        # SINGLE SOURCE: all model specs
│   ├── protocol.py        # fit() / predict() interface
│   ├── cache.py           # ModelCache: TSFM singleton lifecycle
│   ├── ensemble.py        # median/mean aggregation
│   └── adapters/
│       ├── tsfm/          # Thin wrappers (~100 lines each)
│       │   ├── chronos.py
│       │   ├── timesfm.py
│       │   ├── moirai.py
│       │   └── patchtst_fm.py
│       └── baseline/      # Naive, SeasonalNaive (stateless)
│           ├── naive.py
│           └── seasonal.py
│
├── pipeline.py            # ONE FILE: forecast(), run_forecast()
├── inspect.py             # ONE FILE: list_models(), check_health()
└── __init__.py            # Clean public API
```

**Total**: ~2,000 lines core (adapters excluded)

---

## Design Patterns (From Nanobot)

### 1. Registry Pattern (Single Source of Truth)

```python
# models/registry.py - The ONLY place that defines available models

@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter_path: str          # "tsagentkit.models.adapters.tsfm.chronos"
    config_fields: dict        # {"context_length": 512, ...}
    requires: list[str]        # ["chronos", "torch"]
    is_tsfm: bool

REGISTRY: dict[str, ModelSpec] = {
    "chronos": ModelSpec(
        name="chronos",
        adapter_path="tsagentkit.models.adapters.tsfm.chronos",
        config_fields={"context_length": 512},
        requires=["chronos", "torch"],
        is_tsfm=True,
    ),
    "timesfm": ModelSpec(...),
    "moirai": ModelSpec(...),
    "naive": ModelSpec(...),
    "seasonal_naive": ModelSpec(...),
}

def list_models(tsfm_only: bool = False) -> list[str]:
    """Single function to query available models."""
    ...
```

**Benefit**: Add a new TSFM = add one entry to REGISTRY. Done.

### 2. Protocol Over Inheritance

```python
# models/protocol.py - Just functions, no base classes

ModelArtifact = Any  # Adapter decides what to store

def fit(spec: ModelSpec, dataset: TSDataset) -> ModelArtifact:
    """Load adapter dynamically, fit model."""
    adapter = _load_adapter(spec.adapter_path)
    return adapter.fit(dataset, **spec.config_fields)

def predict(
    spec: ModelSpec,
    artifact: ModelArtifact,
    dataset: TSDataset,
    h: int
) -> pd.DataFrame:
    """Generate predictions."""
    adapter = _load_adapter(spec.adapter_path)
    return adapter.predict(artifact, dataset, h)
```

**Benefit**: No complex inheritance trees. Pure functions.

### 3. Single Config Object

```python
# core/config.py - ONE config to rule them all

@dataclass(frozen=True)
class ForecastConfig:
    # Required
    h: int
    freq: str = "D"

    # Ensemble
    ensemble_method: Literal["median", "mean"] = "median"
    min_tsfm: int = 1                    # Min TSFMs required

    # Output
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    quantile_mode: Literal["best_effort", "strict"] = "best_effort"

    @staticmethod
    def quick(h: int, freq: str = "D") -> "ForecastConfig":
        return ForecastConfig(h=h, freq=freq)

    @staticmethod
    def strict(h: int, freq: str = "D") -> "ForecastConfig":
        return ForecastConfig(h=h, freq=freq, min_tsfm=1)
```

`quantile_mode` behavior:
- `"best_effort"`: include `q*` columns when adapters provide them; otherwise continue with point forecast.
- `"strict"`: fail if requested quantiles are unavailable from all model predictions.

**Benefit**: One frozen dataclass. No Pydantic, no validation overhead.

### 4. Pipeline as Simple Function

```python
# pipeline.py - The entire standard pipeline in ONE file

def forecast(
    data: pd.DataFrame,
    h: int,
    freq: str = "D",
    **kwargs
) -> ForecastResult:
    """
    Zero-config TSFM ensemble forecast.
    Core logic: validate → build dataset → fit TSFMs → ensemble → return
    """
    cfg = ForecastConfig(h=h, freq=freq, **kwargs)

    # 1. Validate (2 lines)
    df = _validate(data)

    # 2. Build dataset (3 lines)
    dataset = TSDataset.from_dataframe(df, cfg)

    # 3. Get TSFM models only (1 line)
    models = [m for m in REGISTRY.values() if m.is_tsfm]

    # 4. Fit all in parallel (5 lines)
    artifacts = _fit_all(models, dataset)

    # 5. Predict all (3 lines)
    predictions = _predict_all(models, artifacts, dataset, cfg.h)

    # 6. Ensemble (2 lines)
    result = ensemble(predictions, method=cfg.ensemble_method)

    return ForecastResult(df=result, models_used=[m.name for m in models])
```

**Benefit**: Entire standard pipeline in ~50 lines. Readable. Hackable.

### 5. TSFM Model Caching (Critical for Performance)

**Problem**: TSFMs (Chronos, TimesFM, Moirai) have large parameters (100MB-2GB). Loading on every `forecast()` call is prohibitively expensive.

**Solution**: `ModelCache` with singleton lifecycle management.

```python
# models/cache.py - Model lifecycle management

from typing import Any

class ModelCache:
    """
    Singleton cache for loaded TSFM models.

    TSFMs are expensive to load but cheap to predict.
    Cache keeps them in memory for reuse across calls.
    """

    _cache: dict[str, Any] = {}  # model_name -> loaded_model

    @classmethod
    def get(cls, spec: ModelSpec) -> Any:
        """Get cached model or load if not exists."""
        if spec.name not in cls._cache:
            cls._cache[spec.name] = cls._load(spec)
        return cls._cache[spec.name]

    @classmethod
    def preload(cls, models: list[ModelSpec]) -> None:
        """Pre-load multiple models (useful for batch processing)."""
        for spec in models:
            if spec.name not in cls._cache:
                cls._cache[spec.name] = cls._load(spec)

    @classmethod
    def unload(cls, model_name: str | None = None) -> None:
        """
        Unload model(s) to free memory.

        Args:
            model_name: Specific model to unload, or None to clear all
        """
        names = [model_name] if model_name is not None else list(cls._cache.keys())
        for name in names:
            model = cls._cache.pop(name, None)
            if model is not None:
                cls._unload_adapter(name, model)
        cls._release_backend_memory()

    @classmethod
    def _load(cls, spec: ModelSpec) -> Any:
        """Load model from adapter."""
        adapter = _load_adapter(spec.adapter_path)
        return adapter.load(**spec.config_fields)

    @classmethod
    def _unload_adapter(cls, model_name: str, model: Any) -> None:
        """Best-effort adapter unload hook."""
        ...

    @classmethod
    def _release_backend_memory(cls) -> None:
        """Best-effort gc + backend cache cleanup."""
        ...

    @classmethod
    def list_loaded(cls) -> list[str]:
        """List currently cached models."""
        return list(cls._cache.keys())
```

**Standard Pipeline (Automatic Caching)**:

```python
# pipeline.py - Automatic cache management

def forecast(data: pd.DataFrame, h: int, freq: str = "D", **kwargs) -> ForecastResult:
    cfg = ForecastConfig(h=h, freq=freq, **kwargs)

    # ... validate, build dataset ...

    models = [m for m in REGISTRY.values() if m.is_tsfm]

    # Automatic: Use cached models if available, load if not
    predictions = []
    for spec in models:
        model = ModelCache.get(spec)  # Auto-load or reuse
        pred = predict(spec, model, dataset, cfg.h)
        predictions.append(pred)

    # Cache persists after function returns - next forecast() reuses models
    return ensemble(predictions, method=cfg.ensemble_method)
```

**Agent Building (Explicit Control)**:

```python
from tsagentkit.models.cache import ModelCache
from tsagentkit.models.registry import REGISTRY

# 1. Preload all TSFMs upfront (one-time cost)
models = [m for m in REGISTRY.values() if m.is_tsfm]
ModelCache.preload(models)

# 2. Run many forecasts - models stay in memory
for df in large_dataset_batch:
    result = forecast(df, h=7)  # Uses cached models, no reload

# 3. Optional: Unload when done to free memory
ModelCache.unload()  # Clear all
# Or: ModelCache.unload("chronos")  # Clear specific model
```

**Memory Management Guidelines**:

| Scenario | Strategy |
|----------|----------|
| **Single forecast** | Automatic caching (models stay until process ends) |
| **Batch processing** | `preload()` before loop, `unload()` after |
| **Long-running server** | Keep cached, monitor memory, unload oldest |
| **Memory-constrained** | Process one model at a time, unload immediately |

---

## API Surface (Minimal)

### Public API (top-level exports)

```python
from tsagentkit import (
    # Standard Pipeline (zero config, automatic caching)
    forecast,
    run_forecast,
    ForecastConfig,
    ForecastResult,

    # Agent Building (granular control)
    validate,
    build_dataset,
    make_plan,          # make_plan(tsfm_only=True) -> list[ModelSpec]
    fit_all,
    predict_all,
    ensemble,
    TSDataset,
    CovariateSet,

    # Model Cache Control (explicit lifecycle management)
    ModelCache,

    # Registry and diagnostics
    REGISTRY,
    ModelSpec,
    list_models,        # registry listing (TSFMs are required dependencies)
    check_health,
)
```

### ModelCache API

```python
# Automatic in standard pipeline - no manual intervention needed
result = forecast(df, h=7)  # Models cached automatically

# Explicit control for batch processing
from tsagentkit import ModelCache

# Preload (optional optimization for batch)
ModelCache.preload([REGISTRY["chronos"], REGISTRY["timesfm"]])

# Check status
print(ModelCache.list_loaded())  # ["chronos", "timesfm"]

# Unload when done
ModelCache.unload()  # Clear all
# ModelCache.unload("chronos")  # Or clear specific model
```

`ModelCache.unload()` is best-effort:
- releases tsagentkit-owned references
- calls adapter-level `unload` hooks (if implemented)
- triggers backend cleanup (`gc.collect`, CUDA/MPS cache clear)
- cannot reclaim memory still held by external references

---

## Error Handling (4 Types Only)

```python
# core/errors.py

class TSAgentKitError(Exception):
    """Base error with fix hint."""
    code: str
    hint: str

class EContract(TSAgentKitError):
    """Input data invalid (wrong columns, types, etc.)"""
    code = "E_CONTRACT"
    hint = "Ensure df has columns: unique_id, ds, y"

class ENoTSFM(TSAgentKitError):
    """No TSFM models registered (internal invariant violation)."""
    code = "E_NO_TSFM"
    hint = "TSFM registry invariant violated. Ensure default TSFM specs exist in models.registry.REGISTRY."

class EInsufficient(TSAgentKitError):
    """Not enough TSFMs succeeded."""
    code = "E_INSUFFICIENT"
    hint = "Some TSFMs failed. Check logs or reduce min_tsfm."

class ETemporal(TSAgentKitError):
    """Temporal integrity violation."""
    code = "E_TEMPORAL"
    hint = "Data must be sorted by ds. No future dates in covariates."
```

**Benefit**: 4 errors cover 99% of cases. Each has a fix hint.

---

## TSFM Adapter Reference

### Chronos 2 (Amazon)

**Installation**: `pip install chronos-forecasting pandas pyarrow`

**Usage**:
```python
from tsagentkit.models.adapters.tsfm.chronos import load, predict
from tsagentkit import TSDataset, ForecastConfig

# Load model
model = load(model_name="amazon/chronos-2")  # or amazon/chronos-2-small

# Create dataset
config = ForecastConfig(h=7, freq="D")
dataset = TSDataset.from_dataframe(df, config)

# Predict
forecast_df = predict(model, dataset, h=7)
```

**Direct API** (without tsagentkit):
```python
from chronos import Chronos2Pipeline
import torch

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"  # or "cpu"
)

# Predict quantiles directly
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["y"].values),
    prediction_length=7,
    quantile_levels=[0.1, 0.5, 0.9],
)
```

### TimesFM 2.5 (Google)

**Installation**: `pip install tsagentkit-timesfm`

**Usage**:
```python
from tsagentkit.models.adapters.tsfm.timesfm import load, predict
from tsagentkit import TSDataset, ForecastConfig

# Load model
model = load()  # Loads google/timesfm-2.5-200m-pytorch

# Create dataset
config = ForecastConfig(h=7, freq="D")
dataset = TSDataset.from_dataframe(df, config)

# Predict
forecast_df = predict(model, dataset, h=7)
```

**Direct API** (without tsagentkit):
```python
import timesfm
import numpy as np

# Load model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Configure
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    )
)

# Forecast
point_forecast, quantile_forecast = model.forecast(
    horizon=7,
    inputs=[np.array(df["y"].values)],
    freq=[0],  # 0=high freq (daily), 1=medium, 2=low
)
```

### Moirai 2.0 (Salesforce)

**Installation**: `pip install tsagentkit-uni2ts`

**Usage**:
```python
from tsagentkit.models.adapters.tsfm.moirai import load, predict
from tsagentkit import TSDataset, ForecastConfig

# Load model
model = load(model_name="Salesforce/moirai-2.0-R-small")

# Create dataset
config = ForecastConfig(h=7, freq="D")
dataset = TSDataset.from_dataframe(df, config)

# Predict
forecast_df = predict(model, dataset, h=7)
```

**Direct API** (without tsagentkit):
```python
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from gluonts.dataset.pandas import PandasDataset
import pandas as pd

# Load model
module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

# Create forecaster
forecaster = Moirai2Forecast(
    module=module,
    prediction_length=7,
    context_length=1680,  # Moirai 2.0 recommended context
    target_dim=1,  # Univariate
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)

# Create predictor
predictor = forecaster.create_predictor(batch_size=32)

# Prepare data (GluonTS format)
ts_df = pd.DataFrame({
    "target": df["y"].values,
}, index=pd.date_range(start=df["ds"].iloc[0], periods=len(df), freq="D"))

gts_dataset = PandasDataset([{
    "target": ts_df["target"].values,
    "start": ts_df.index[0]
}])

# Predict
forecasts = list(predictor.predict(gts_dataset))
median_forecast = forecasts[0].quantile(0.5)
```

### PatchTST-FM-r1 (IBM)

**Installation**: `pip install tsagentkit-patchtst-fm>=1.0.2`

**Usage**:
```python
from tsagentkit.models.adapters.tsfm.patchtst_fm import load, predict
from tsagentkit import TSDataset, ForecastConfig

# Load model (auto-selects device: cuda > cpu, mps guarded)
model = load(model_name="ibm-research/patchtst-fm-r1")

# Create dataset
config = ForecastConfig(h=7, freq="D")
dataset = TSDataset.from_dataframe(df, config)

# Predict (returns yhat and requested quantiles when available)
forecast_df = predict(model, dataset, h=7)
```

**Direct API** (without tsagentkit):
```python
from tsfm_public import PatchTSTFMForPrediction
import torch
import numpy as np

# Determine device (defaults to cpu on mps for stability)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = PatchTSTFMForPrediction.from_pretrained(
    "ibm-research/patchtst-fm-r1",
    device_map=device,
)
model.eval()

# Prepare context
context = np.asarray(df["y"].values, dtype=np.float32)
context_tensor = torch.tensor(context, dtype=torch.float32, device=device)

# Generate forecast with quantiles
with torch.no_grad():
    outputs = model(
        inputs=[context_tensor],
        prediction_length=7,
        quantile_levels=[0.5],  # Median forecast
        return_loss=False,
    )
    # Extract predictions from outputs
    predictions = outputs.prediction_outputs  # Shape: (batch, quantiles, horizon)

# Get median forecast
median_forecast = predictions[0, 0, :].cpu().numpy()  # First batch, median quantile
```

---

## Adding a New TSFM Adapter (3 Steps)

### Step 1: Create Adapter (~100 lines)

```python
# models/adapters/tsfm/mytsfm.py
"""MyTSFM adapter - ~100 lines.

TSFMs are stateless (pre-trained). The adapter implements:
- load(): Load model weights (called by ModelCache)
- predict(): Generate predictions using loaded model
"""

import pandas as pd
from typing import Any
from tsagentkit.core.dataset import TSDataset

def load(model_name: str = "my-model") -> Any:
    """Load model weights. Called by ModelCache on first use."""
    from mytsfm_library import MyTSFMModel
    return MyTSFMModel.from_pretrained(model_name)

def unload(model: Any | None = None) -> None:
    """Best-effort adapter unload hook. Called by ModelCache.unload()."""
    del model

def fit(dataset: TSDataset) -> Any:
    """TSFMs are pre-trained, no fitting needed."""
    return load()

def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate predictions using loaded model."""
    forecasts = []

    for unique_id in dataset.df["unique_id"].unique():
        mask = dataset.df["unique_id"] == unique_id
        series_df = dataset.df[mask].sort_values("ds")
        context = series_df["y"].values

        # Generate forecast
        prediction = model.predict(context, h)

        # Create forecast DataFrame
        last_date = series_df["ds"].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=dataset.config.freq)[1:]

        forecast_df = pd.DataFrame({
            "unique_id": unique_id,
            "ds": future_dates,
            "yhat": prediction,
        })
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
```

### Step 2: Register in Registry (1 line)

```python
# models/registry.py

REGISTRY = {
    # ... existing models ...
    "mytsfm": ModelSpec(
        name="mytsfm",
        adapter_path="tsagentkit.models.adapters.tsfm.mytsfm",
        config_fields={"context_length": 512},
        requires=["mytsfm", "torch"],
        is_tsfm=True,
    ),
}
```

### Step 3: Test (optional but recommended)

```python
# tests/models/adapters/test_mytsfm.py

from tsagentkit.models.cache import ModelCache
from tsagentkit.models.registry import REGISTRY

def test_mytsfm_basic():
    spec = REGISTRY["mytsfm"]
    model = ModelCache.get(spec)  # Auto-loads
    result = predict(model, sample_dataset, h=7)
    assert len(result) == 7

def test_mytsfm_cache_reuse():
    """Models are cached and reused."""
    spec = REGISTRY["mytsfm"]
    model1 = ModelCache.get(spec)
    model2 = ModelCache.get(spec)
    assert model1 is model2  # Same instance

def test_mytsfm_unload():
    """Unload frees memory."""
    ModelCache.unload("mytsfm")
    assert "mytsfm" not in ModelCache.list_loaded()
```

**Done.** The new TSFM is automatically available via `forecast()` with caching.

---

## Development Workflow

### Environment

```bash
# Setup
uv sync --all-extras

# Run tests
uv run pytest

# Type check
uv run mypy src/tsagentkit

# Format
uv run ruff format src/
```

### Testing Philosophy

- **Unit tests**: Individual functions
- **Integration tests**: Full pipeline
- **Adapter tests**: Mock + real (`TSFM_RUN_REAL=1`)
- **Property tests**: Ensemble aggregation correctness

### Code Style

```python
# Good: Pure function, clear types, docstring
def ensemble(
    predictions: list[pd.DataFrame],
    method: Literal["median", "mean"] = "median"
) -> pd.DataFrame:
    """
    Element-wise ensemble of predictions.

    Args:
        predictions: List of DataFrames with columns [unique_id, ds, yhat]
        method: Aggregation method

    Returns:
        DataFrame with ensemble predictions
    """
    if not predictions:
        raise EInsufficient("No predictions to ensemble")
    # ... implementation ...
```

---

## Comparison: v1.x vs v2.0 (Nanobot Style)

| Aspect | v1.x (Legacy) | v2.0 (Nanobot-Inspired) |
|--------|---------------|------------------------|
| **Lines of Code** | ~10,000+ | ~2,000 core |
| **Config Objects** | 5+ (TaskSpec, PlanSpec, RouterConfig, ...) | 1 (ForecastConfig) |
| **Pipeline** | 12+ stages | 6 stages |
| **Model Selection** | Competitive backtest | Ensemble only |
| **Registry** | Multiple registries | Single REGISTRY dict |
| **Errors** | 20+ types | 4 types |
| **API Surface** | 30+ functions | 8 pipeline primitives (+ minimal core types) |
| **Adapters** | Complex inheritance | Simple functions |
| **Add TSFM** | 5 files, 50 lines | 2 files, 100 lines |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dataclass over Pydantic** | Faster, no JSON schema overhead |
| **Registry pattern** | Single source of truth for models |
| **Protocol over inheritance** | Simpler, more flexible |
| **TSFM-only ensemble** | Cleaner production contract |
| **Pure functions** | Easier to test, reason about |
| **Frozen configs** | Immutable, hashable, safe |
| **Lazy loading** | Fast startup when TSFM not used |

---

## Quick Reference

### Standard Pipeline

```python
from tsagentkit import forecast

# Zero config, TSFM ensemble
result = forecast(df, h=7, freq="D")
print(result.df)
```

### Agent Building

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

# Custom pipeline
config = ForecastConfig(h=7, freq="D")
df = validate(raw_df)
dataset = build_dataset(df, config)
models = make_plan(tsfm_only=True)

# Fit & predict
artifacts = fit_all(models, dataset)
preds = predict_all(models, artifacts, dataset, h=config.h, quantiles=config.quantiles)

# Ensemble
result = ensemble(
    preds,
    method=config.ensemble_method,
    quantiles=config.quantiles,
    quantile_mode=config.quantile_mode,
)
```

### Inspection

```python
from tsagentkit import list_models, check_health

# Registry listing (all registered TSFMs)
print(list_models(tsfm_only=True))
# ['chronos', 'timesfm', 'moirai', 'patchtst_fm']

# Health check
health = check_health()
print(health.tsfm_available)
print(health.tsfm_missing)    # [] under TSFM-required contract
```

### Model Cache (Batch Processing)

```python
from tsagentkit import forecast, ModelCache
from tsagentkit.models.registry import REGISTRY

# Scenario: Processing 1000 time series

# Option 1: Automatic caching (simplest)
for df in dataset_batch:
    result = forecast(df, h=7)  # Models loaded once, reused automatically

# Option 2: Explicit preload (better for memory control)
models = [m for m in REGISTRY.values() if m.is_tsfm]
ModelCache.preload(models)  # Load all TSFMs upfront

for df in dataset_batch:
    result = forecast(df, h=7)  # Uses preloaded models

ModelCache.unload()  # Free memory when done

# Check cache status
print(ModelCache.list_loaded())  # ['chronos', 'timesfm', 'moirai', 'patchtst_fm']
```

---

## Summary

`tsagentkit` v2.0 embraces **nanobot's minimalist philosophy** with production-grade optimizations:

- **~2,000 lines** for full TSFM ensemble capability
- **Single registry** for all models
- **One config** object
- **8 pipeline primitives** (`forecast`, `run_forecast`, `validate`, `build_dataset`, `make_plan`, `fit_all`, `predict_all`, `ensemble`)
- **4 error types**
- **ModelCache** for efficient TSFM lifecycle management (load once, reuse many)

Research-ready. Hackable. Production-grade.

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away." — Antoine de Saint-Exupéry
