# tsagentkit Documentation

Deterministic execution engine for time-series forecasting, designed to be called by coding agents.

Use this page as the documentation index.

## Start Here

Read in this order:
1. [Main README](../README.md) - quick install and runnable examples
2. [PRD](PRD.md) - contract-level requirements and guardrails
3. [Architecture](ARCHITECTURE.md) - module boundaries and system design
4. [API Stability](API_STABILITY.md) - compatibility guarantees

## Use Cases

### Agent Integration

- Start with `forecast(...)` for the fastest integration.
- Move to `run_forecast(...)` for full orchestration and run artifacts.
- Use assembly-first APIs when you need step-level control.
- Agent-facing docs:
- [skill/QUICKSTART.md](../skill/QUICKSTART.md)
- [skill/README.md](../skill/README.md)
- [skill/recipes.md](../skill/recipes.md)
- [skill/tool_map.md](../skill/tool_map.md)
- [skill/TROUBLESHOOTING.md](../skill/TROUBLESHOOTING.md)

### Production Operation

- Use `TaskSpec.production(...)` and `run_forecast(...)`.
- Persist and validate artifacts before serving with:
- `save_run_artifact`
- `load_run_artifact`
- `validate_run_artifact_for_serving`
- `replay_forecast_from_artifact`
- Review:
- [API Stability](API_STABILITY.md)
- [Release v1.1 Checklist](RELEASE_V1_1.md)

### Repository Contribution

- Use `uv` for environment and command execution.
- See [AGENTS.md](../AGENTS.md) for contribution rules and test commands.

## API Surfaces

### Top-level user-facing API

```python
from tsagentkit import (
    TaskSpec,
    forecast,
    diagnose,
    repair,
    run_forecast,
    validate_contract,
    run_qa,
    build_dataset,
    make_plan,
    fit,
    predict,
    package_run,
    save_run_artifact,
    load_run_artifact,
    validate_run_artifact_for_serving,
    replay_forecast_from_artifact,
    describe,
)
```

### Hierarchy

```python
from tsagentkit.hierarchy import (
    HierarchyStructure,
    Reconciler,
    ReconciliationMethod,
    reconcile_forecasts,
)
```

### TSFM adapters (optional, `tsagentkit[tsfm]`)

```python
from tsagentkit.models.adapters import (
    AdapterConfig,
    AdapterRegistry,
)

available = AdapterRegistry.list_available()
print(available)
```

### Serving submodule

```python
from tsagentkit.serving import (
    MonitoringConfig,
    run_forecast,
    package_run,
    save_run_artifact,
    load_run_artifact,
    validate_run_artifact_for_serving,
    replay_forecast_from_artifact,
    TSFMModelCache,
    get_tsfm_model,
    clear_tsfm_cache,
)
```

### Features submodule

```python
from tsagentkit.features import FeatureConfig, FeatureFactory
```

## Canonical Execution Flow

`validate_contract` -> `run_qa` -> `build_dataset` -> `make_plan` -> `fit` -> `predict` -> `package_run`

For one-call orchestration, use `run_forecast(...)`.

## Developer Commands

```bash
uv sync
uv run pytest
TSFM_RUN_REAL=1 uv run pytest -m tsfm
uv run lint-imports
```

## Additional References

- [ADR 001: Assembly-First Integration](ADR_001_assembly_first.md)
- [Release v1.1 Checklist](RELEASE_V1_1.md)
- Historical planning notes: `docs/v1.1.1/`
