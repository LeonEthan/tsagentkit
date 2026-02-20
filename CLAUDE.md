# CLAUDE.md

> **Design Philosophy**: Minimalist, research-ready time-series forecasting toolkit for vibe coding agents.
> **Core Tenet**: Less code, more clarity. ~2,000 lines for full TSFM ensemble capability.

## Purpose

This file is the compact, high-signal operating guide for agents.

Detailed design notes, architecture rationale, adapter deep dives, and extended examples were moved to:

- `docs/DESIGN.md`

If you need implementation detail, open `docs/DESIGN.md`. If you need to get work done quickly, start here.

---

## Project Overview

`tsagentkit` is an ultra-lightweight execution engine for time-series forecasting. It provides **zero-config TSFM ensemble** for production use, while exposing **granular pipeline functions** for agent customization.

**Version**: 2.0.2  
**Core Lines**: ~2,000 (excluding adapters)  
**Design Goal**: Research-ready code that's easy to understand, modify, and extend.

---

## Core Principles

1. **Radical Simplicity**: Every module has a single, clear responsibility.
2. **Flat Structure**: Minimal nesting and predictable imports.
3. **Pure Functions**: Functional composition over class hierarchies.
4. **Single Source of Truth**: One registry and one config system.
5. **Research-Ready**: Clean code that is easy to modify and extend.

---

## Architecture Snapshot

```
src/tsagentkit/
├── core/
│   ├── config.py
│   ├── dataset.py
│   ├── device.py
│   ├── results.py
│   ├── types.py
│   └── errors.py
├── models/
│   ├── registry.py
│   ├── protocol.py
│   ├── cache.py
│   ├── ensemble.py
│   └── adapters/
│       ├── tsfm/
│       └── baseline/
├── pipeline.py
├── inspect.py
└── __init__.py
```

- `models/registry.py` is the canonical model registry.
- `core/config.py` defines the main forecasting config object.
- `models/cache.py` manages TSFM load/reuse lifecycle.
- `pipeline.py` contains the standard pipeline orchestration.

---

## Usage Modes

### 1. Standard Pipeline (Zero Config)

Use when you want a fast, default TSFM ensemble:

```python
from tsagentkit import forecast

result = forecast(df, h=7, freq="D")
print(result.df)
```

### 2. Agent Building (Full Control)

Use when you need custom planning, execution, or diagnostics:

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
preds = predict_all(models, artifacts, dataset, h=config.h)
result = ensemble(preds, method=config.ensemble_method, quantiles=config.quantiles)
```

---

## Public API (Minimal)

```python
from tsagentkit import (
    # Standard pipeline
    forecast,
    run_forecast,
    ForecastConfig,
    ForecastResult,
    RunResult,

    # Building blocks
    validate,
    build_dataset,
    make_plan,
    fit_all,
    predict_all,
    ensemble,
    TSDataset,
    CovariateSet,

    # Model lifecycle control
    ModelCache,

    # Device resolution
    resolve_device,

    # Registry and diagnostics
    REGISTRY,
    ModelSpec,
    list_models,
    check_health,
)
```

Detailed behavior and examples: `docs/DESIGN.md`.

---

## Model Cache Guidelines

`ModelCache` avoids expensive TSFM reloads across forecasts.

```python
from tsagentkit import forecast, ModelCache
from tsagentkit.models.registry import REGISTRY

# Automatic caching in forecast()
result = forecast(df, h=7)

# Optional preload for large batch jobs
models = [m for m in REGISTRY.values() if m.is_tsfm]
ModelCache.preload(models)

for item in dataset_batch:
    _ = forecast(item, h=7)

ModelCache.unload()  # best-effort memory release
```

`ModelCache.unload()` is best-effort:

- releases tsagentkit-owned references
- calls adapter `unload` hooks if provided
- triggers backend cleanup (`gc.collect`, CUDA/MPS cache clear)
- cannot reclaim memory still held by external references

---

## Error Contract

Use these errors as the stable contract surface:

- `EContract`: input schema/type/contract violations
- `ENoTSFM`: TSFM registry invariant violation (internal misconfiguration)
- `EInsufficient`: too few successful TSFM predictions
- `ETemporal`: temporal integrity violations

See `docs/DESIGN.md` for code/hints and full examples.

---

## Development Workflow

### Environment

```bash
uv sync --all-extras
```

### Testing Philosophy & Commands (Run before PR)

**Quality gates** (must pass before submitting):

```bash
# 1. Lint check & auto-fix (prevents CI ruff failures)
uv run ruff check src/ --fix

# 2. Code formatting (prevents CI format failures)
uv run ruff format src/

# 3. Type checking (prevents CI mypy failures)
uv run mypy src/tsagentkit

# 4. Run tests (prevents CI test failures)
uv run pytest --cov=src/tsagentkit -v
```

**Note**: These mirror CI exactly. If all pass locally, CI will likely pass too.

**Test coverage principles**:
- Unit tests for individual functions
- Integration tests for full pipeline behavior
- Adapter tests with mock mode and optional real mode (`TSFM_RUN_REAL=1`)
- Property checks for ensemble aggregation correctness

### Real TSFM Smoke (Live Adapters)

Run this when you need to validate real backend loading/inference instead of mock-mode behavior:

```bash
TSFM_RUN_REAL=1 uv run pytest tests/ci/test_real_tsfm_smoke_gate.py tests/ci/test_standard_pipeline_real_smoke.py
```

---

## Adding a New TSFM Adapter (Short Path)

1. Add adapter module in `src/tsagentkit/models/adapters/tsfm/` with `load`, `fit`, `predict`, and optional `unload`.
2. Register model once in `src/tsagentkit/models/registry.py`.
3. Add adapter tests in `tests/models/adapters/`.

For full reference implementation and examples, see `docs/DESIGN.md`.

---

## Inspection Quick Commands

```python
from tsagentkit import list_models, check_health

print(list_models(tsfm_only=True))

health = check_health()
print(health.tsfm_available)
print(health.tsfm_missing)
print(health.baselines_available)  # optional baseline adapters only
```

---

## Summary

`AGENTS.md` is now intentionally compact. Use it for execution and daily coding tasks.

Use `docs/DESIGN.md` for:

- detailed architecture rationale
- full design patterns and code sketches
- adapter-specific references (Chronos, TimesFM, Moirai, PatchTST-FM)
- deeper extension guidance

This split keeps agent context cost lower while preserving full project design detail.
