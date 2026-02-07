# tsagentkit Documentation

Complete documentation for the tsagentkit assembly-first time-series forecasting toolkit.

## Getting Started

- [Main README](../README.md) - Overview and quick start
- [Installation](#installation) - Setup instructions
- [Quick Start Guide](#quick-start) - First steps
- [ADR 001: Assembly-First Integration](ADR_001_assembly_first.md) - Product direction and API posture

## Quick Start

- Primary integration path: assembly-first step composition from `README.md`
- Compatibility path: `run_forecast(...)` convenience wrapper when step-level control is not needed

## TSFM Adapters

Adapters are available under `src/tsagentkit/models/adapters/` with
configuration via `AdapterConfig`. Refer to the module docstrings for
model-specific options.

## Hierarchical Forecasting

Reconciliation uses `hierarchicalforecast` with `S_df` and `tags`
as the canonical hierarchy inputs.

## Agent Recipes

Agent-facing recipes live in `skill/recipes.md` and `skill/README.md`.
Package-distributed mirrors are kept in `src/tsagentkit/skill/` and should stay byte-for-byte consistent.

## API Reference

### Stability & Compatibility

- [Stable API Contract](API_STABILITY.md) - Backward-compatibility guarantees
- [v1.1 Release Checklist & Migration Note](RELEASE_V1_1.md) - Release-blocker checks and migration guidance for TSFM-required default and CI gate updates

### Core Components

```python
from tsagentkit import (
    TaskSpec,
    align_covariates,
    build_dataset,
    fit,
    make_plan,
    package_run,
    predict,
    run_qa,
    validate_contract,
)
```

### Hierarchical

```python
from tsagentkit.hierarchy import (
    HierarchyStructure,
    Reconciler,
    ReconciliationMethod,
    reconcile_forecasts,
)
```

### TSFM Adapters

```python
from tsagentkit.models.adapters import (
    ChronosAdapter,
    MoiraiAdapter,
    TimesFMAdapter,
    AdapterRegistry,
)
```

### Serving

```python
from tsagentkit.serving import (
    TSFMModelCache,
    run_forecast,  # convenience wrapper
    package_run,
    save_run_artifact,
    load_run_artifact,
    validate_run_artifact_for_serving,
    replay_forecast_from_artifact,
    get_tsfm_model,
    clear_tsfm_cache,
)
```

## Feature Engineering

`FeatureFactory` supports multiple feature backends via `FeatureConfig.engine`:

- `auto` (default): use `tsfeatures` if available, otherwise fallback to native features.
- `tsfeatures`: use `tsfeatures` for statistical features (requires `tsfeatures`).
- `native`: use the built-in point-in-time safe feature generator (legacy).

Key configuration fields:

- `tsfeatures_features`: list of `tsfeatures` function names to apply.
- `tsfeatures_freq`: optional season length for `tsfeatures`.
- `tsfeatures_dict_freqs`: optional mapping of pandas freq to season length.
- `allow_fallback`: allow auto fallback to native when `tsfeatures` is unavailable.

Example:

```python
from tsagentkit.features import FeatureConfig, FeatureFactory

config = FeatureConfig(
    engine="tsfeatures",
    tsfeatures_features=["acf_features", "stl_features"],
    tsfeatures_freq=7,
    known_covariates=["holiday"],
)

factory = FeatureFactory(config)
matrix = factory.create_features(dataset)
```

## Best Practices

### 1. Model Selection

- Start with smaller models for prototyping
- Use `get_tsfm_model()` for caching in production
- Configure fallback ladder for robustness

### 2. Hierarchical Data

- Validate S-matrix before forecasting
- Use MinT for maximum accuracy
- Check coherence scores

### 3. Performance

- Use GPU when available
- Batch process multiple series
- Cache models in serving environment

### 4. Guardrails

- Never disable temporal validation
- Handle `ESplitRandomForbidden` properly
- Monitor for data leakage

## Version History

### v1.1 - TSFM-First Release Gate (Current)

- TSFM policy default set to required (`TSFMPolicy.mode="required"`)
- Deterministic TSFM policy matrix tests in CI
- Real non-mock TSFM adapter smoke gate in CI
- Release checklist and migration note published (`RELEASE_V1_1.md`)

### v1.0 - Ecosystem

- TSFM adapters (Chronos, Moirai, TimesFM)
- Hierarchical reconciliation (6 methods)
- Model caching for serving
- Complete documentation and recipes

### v0.2 - Enhanced Robustness

- Drift detection
- Retrain triggers
- Stability metrics

### v0.1 - Minimum Loop

- Core pipeline
- Baseline models
- Backtesting

## Contributing

See [AGENTS.md](../AGENTS.md) for contribution guidelines.

## Architecture Checks

To validate dependency boundaries during refactors:

```bash
uv run lint-imports
```

## Support

- GitHub Issues: Bug reports and feature requests
- Recipes: Check `skill/recipes.md` for common scenarios
- API Docs: See module docstrings
