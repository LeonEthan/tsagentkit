# tsagentkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A robust execution engine for AI agents performing time-series forecasting. It is designed as an assembly-first toolkit: coding agents compose production workflows from modular building blocks, while `run_forecast()` remains a convenience wrapper.

## Features

- **TSFM-First Strategy**: Priority to Chronos, Moirai, and TimesFM with automatic fallback ladder
- **Hierarchical Reconciliation**: Full support for hierarchical time series with 6 reconciliation methods
- **Strict Guardrails**: Prevents data leakage, enforces temporal integrity, bans random splits
- **Model Caching**: Efficient TSFM model caching for production serving
- **Provenance Tracking**: Complete audit trail with signatures for reproducibility
- **Monitoring**: Drift detection and automated retrain triggers

## v1.0 Feature Matrix

| Category | Feature | Status | Notes |
|----------|---------|--------|-------|
| **Core** | Input validation | ✅ | `validate_contract()` with schema checking |
| | Task specification | ✅ | `TaskSpec` with h, freq, quantiles |
| | Deterministic signatures | ✅ | `TaskSpec.model_hash()` for reproducibility |
| **QA** | Data quality checks | ✅ | `run_qa()` with multiple modes |
| | Leakage detection | ✅ | Future covariate leakage protection |
| | Auto-repair | ✅ | Repair strategies for common issues |
| **Series** | TSDataset | ✅ | Immutable dataset with validation |
| | Sparsity detection | ✅ | Regular, intermittent, cold-start, sparse |
| | Time alignment | ✅ | Resampling and temporal operations |
| **Routing** | Auto model selection | ✅ | Based on data characteristics |
| | Fallback ladder | ✅ | TSFM → Baselines → Naive |
| | Intermittent handling | ✅ | Croston method for sparse demand |
| | Bucketing (v0.2) | ✅ | Data bucketing for complex routing |
| **Models** | Baseline models | ✅ | SeasonalNaive, ETS, Theta, HistoricAverage, etc. |
| | Chronos adapter | ✅ | Amazon TSFM support |
| | Moirai adapter | ✅ | Salesforce TSFM support |
| | TimesFM adapter | ✅ | Google TSFM support |
| | Model caching | ✅ | `TSFMModelCache` for efficient loading |
| **Backtest** | Rolling windows | ✅ | Expanding and sliding strategies |
| | Temporal integrity | ✅ | Strict ordering validation |
| | Segment diagnostics | ✅ | Metrics by sparsity class |
| | Temporal diagnostics | ✅ | Hour/day error patterns |
| **Hierarchical** | Structure definition | ✅ | `HierarchyStructure` with S-matrix |
| | 6 reconciliation methods | ✅ | Bottom-up, top-down, OLS, WLS, MinT, etc. |
| | Auto reconciliation | ✅ | Available in assembly and wrapper pipelines |
| | Coherence validation | ✅ | `is_coherent()` checks |
| **Serving** | `package_run()` | ✅ | Assembly-first artifact packaging |
| | `run_forecast()` | ✅ | Convenience pipeline wrapper |
| | Mode support | ✅ | quick, standard, strict |
| | Feature engineering | ✅ | Optional `FeatureConfig` integration |
| | Structured logging | ✅ | JSON event logging with `StructuredLogger` |
| **Monitoring** | Drift detection | ✅ | PSI and Kolmogorov-Smirnov methods |
| | Stability monitoring | ✅ | Prediction jitter detection |
| | Retrain triggers | ✅ | Automated trigger evaluation |
| **Provenance** | Data signatures | ✅ | SHA-256 based data hashing |
| | Config signatures | ✅ | Deterministic config hashing |
| | Full traceability | ✅ | Complete audit trail in `Provenance` |
| **Skill** | Agent documentation | ✅ | `skill/README.md` with module guide |
| | Tool map | ✅ | Complete API reference |
| | Recipes | ✅ | 8 runnable end-to-end examples |
| **Errors** | Structured errors | ✅ | All errors with codes and context |
| | Guardrails | ✅ | `E_SPLIT_RANDOM_FORBIDDEN`, `ECovariateLeakage` |

## Installation

TSFM-first runtime (Chronos/Moirai/TimesFM adapters) is included in the default package:

```bash
pip install tsagentkit
```

Or with uv:

```bash
uv pip install tsagentkit
```

## Testing

Run the test suite:

```bash
uv run pytest
```

Run real TSFM smoke tests (downloads models):

```bash
TSFM_RUN_REAL=1 uv run pytest -m tsfm
```

Run the minimal non-mock CI smoke locally:

```bash
TSFM_RUN_REAL=1 uv run pytest -v tests/ci/test_real_tsfm_smoke_gate.py
```

## Quick Start

### Assembly-First (Recommended)

```python
import pandas as pd
from tsagentkit import (
    TaskSpec,
    align_covariates,
    build_dataset,
    fit,
    make_plan,
    package_run,
    predict,
    rolling_backtest,
    run_qa,
    validate_contract,
)

# Prepare data
df = pd.DataFrame({
    "unique_id": ["A"] * 30,
    "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
    "y": range(30),
})

# Create task spec
spec = TaskSpec(h=7, freq="D")

# 1) Contract validation
report = validate_contract(df)
report.raise_if_errors()

# 2) QA checks
qa_report = run_qa(df, spec, mode="standard")

# 3) Covariate alignment (works for no-covariate panels too)
aligned = align_covariates(df, spec)

# 4) Dataset build
dataset = build_dataset(aligned.panel, spec, validate=False)
dataset = dataset.with_covariates(aligned, panel_with_covariates=df)

# 5) Deterministic routing plan
plan, route_decision = make_plan(dataset, spec)

# 6) Optional backtest for selection diagnostics
backtest_report = rolling_backtest(dataset=dataset, spec=spec, plan=plan)

# 7) Fit + predict
model_artifact = fit(dataset, plan, covariates=aligned)
forecast = predict(dataset, model_artifact, spec, covariates=aligned)

# 8) Package full run artifact
artifact = package_run(
    forecast=forecast,
    plan=plan,
    task_spec=spec.model_dump(),
    backtest_report=backtest_report,
    qa_report=qa_report,
    model_artifact=model_artifact,
    provenance=forecast.provenance,
    metadata={"route_decision": route_decision.model_dump()},
)

print(artifact.forecast.df.head())
```

### Convenience Wrapper (`run_forecast`)

```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec(h=7, freq="D")
artifact = run_forecast(df, spec, mode="standard")
print(artifact.forecast.df.head())
```

### Hierarchical Forecasting

```python
import numpy as np
from tsagentkit import make_plan
from tsagentkit.hierarchy import (
    HierarchyStructure,
    ReconciliationMethod,
    reconcile_forecasts,
)

# Define hierarchy
hierarchy = HierarchyStructure(
    aggregation_graph={"Total": ["A", "B"]},
    bottom_nodes=["A", "B"],
    s_matrix=np.array([
        [1, 0],  # A
        [0, 1],  # B
        [1, 1],  # Total = A + B
    ]),
)

# Attach to dataset
dataset = dataset.with_hierarchy(hierarchy)

# Forecast with automatic reconciliation
plan, route_decision = make_plan(dataset, spec)
```

### Using TSFM Models

```python
from tsagentkit.serving import get_tsfm_model

# Load cached TSFM model
adapter = get_tsfm_model("chronos", model_size="base")

# Generate forecast
result = adapter.predict(dataset, horizon=spec.horizon)
```

### Agent Assembly Primitives (Phase 4)

```python
from tsagentkit import (
    attach_plan_graph,
    build_plan_graph,
    get_adapter_capability,
    list_adapter_capabilities,
    make_plan,
)

# Build a deterministic routing plan
plan, route_decision = make_plan(dataset, spec)

# Expose execution as explicit DAG-style nodes for custom agent orchestration
graph = build_plan_graph(plan, include_backtest=True)
print(graph.entrypoints, graph.terminal_nodes)

# Or attach graph metadata directly onto PlanSpec for downstream steps
plan_with_graph = attach_plan_graph(plan, include_backtest=True)

# Inspect adapter capabilities and runtime availability
capabilities = list_adapter_capabilities()
chronos = get_adapter_capability("chronos")
print(chronos.available, chronos.supports_future_covariates)
```

### Artifact Lifecycle (Phase 5)

```python
from tsagentkit import (
    load_run_artifact,
    replay_forecast_from_artifact,
    save_run_artifact,
    validate_run_artifact_for_serving,
)

# Persist artifact as versioned JSON payload
save_run_artifact(artifact, "artifacts/run_artifact.json")

# Load + schema-compat validate
loaded = load_run_artifact("artifacts/run_artifact.json")
report = validate_run_artifact_for_serving(
    loaded,
    expected_task_signature=loaded.provenance.task_signature,
    expected_plan_signature=loaded.provenance.plan_signature,
)

# Deterministic forecast replay from artifact payload
replayed_forecast = replay_forecast_from_artifact(loaded)
print(report["artifact_schema_version"], replayed_forecast.df.head())
```

## Architecture

```
Input Data
    │
    ▼
┌─────────────┐
│  Validation │── Guardrails (E_SPLIT_RANDOM_FORBIDDEN)
└─────────────┘
    │
    ▼
┌─────────────┐
│     QA      │── Leakage detection, data quality
└─────────────┘
    │
    ▼
┌─────────────┐
│   Router    │── TSFM selection, fallback ladder
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│  Backtest   │────▶│  reconcile  │ (if hierarchical)
└─────────────┘     └─────────────┘
    │
    ▼
┌─────────────┐
│    Fit      │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│   Predict   │────▶│  reconcile  │ (if hierarchical)
└─────────────┘     └─────────────┘
    │
    ▼
┌─────────────┐
│   Package   │── Provenance, signatures
└─────────────┘
```

## TSFM Adapters

tsagentkit provides unified adapters for major Time-Series Foundation Models:

### Chronos (Amazon)
```python
from tsagentkit.models.adapters import AdapterConfig, ChronosAdapter

config = AdapterConfig(model_name="chronos", model_size="base")
adapter = ChronosAdapter(config)
```

### Moirai (Salesforce)
```python
from tsagentkit.models.adapters import AdapterConfig, MoiraiAdapter

config = AdapterConfig(model_name="moirai", model_size="base")
adapter = MoiraiAdapter(config)
```

### TimesFM (Google)
```python
from tsagentkit.models.adapters import AdapterConfig, TimesFMAdapter

config = AdapterConfig(model_name="timesfm", model_size="base")
adapter = TimesFMAdapter(config)
```

See `docs/README.md` and adapter docstrings for configuration details.

### TSFM Routing Policy (Phase 1)

Use `TaskSpec.tsfm_policy` to make TSFM requirements explicit:

```python
from tsagentkit import TaskSpec

# Default: require TSFM; fail fast if no preferred adapter is available
spec = TaskSpec(h=14, freq="D")

# Relax policy explicitly if baseline fallback is desired
preferred_tsfm_spec = TaskSpec(
    h=14,
    freq="D",
    tsfm_policy={
        "mode": "preferred",
        "adapters": ["chronos", "moirai", "timesfm"],
    },
)

# Disable TSFM routing explicitly
no_tsfm_spec = TaskSpec(
    h=14,
    freq="D",
    tsfm_policy={"mode": "disabled"},
)
```

## Hierarchical Reconciliation

6 reconciliation methods available:

| Method | Description | Best For |
|--------|-------------|----------|
| Bottom-Up | Aggregate bottom forecasts | Bottom patterns important |
| Top-Down | Distribute top forecast | Top patterns reliable |
| Middle-Out | Forecast at middle level | Deep hierarchies |
| OLS | Least squares optimal | Balanced approach |
| WLS | Weighted least squares | Different variances |
| MinT | Minimum trace (optimal) | Maximum accuracy |

```python
from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

reconciled = reconcile_forecasts(
    base_forecasts=forecasts,
    structure=hierarchy,
    method=ReconciliationMethod.MIN_TRACE,
)
```

See `docs/ARCHITECTURE.md` for hierarchy contract details (`S_df`/`tags`).

## Guardrails

tsagentkit enforces strict time-series best practices:

- **E_SPLIT_RANDOM_FORBIDDEN**: Random train/test splits are banned
- **E_COVARIATE_LEAKAGE**: Future covariate leakage detection
- **Temporal Integrity**: Data must be temporally ordered

```python
from tsagentkit.contracts import ESplitRandomForbidden

try:
    dataset = TSDataset.from_dataframe(shuffled_df, spec)
except ESplitRandomForbidden as e:
    print(f"Guardrail triggered: {e}")
```

## Agent Recipes

Pre-built recipes for AI agents:

- `skill/recipes.md`
- `skill/README.md`

## Development

### Setup

```bash
git clone https://github.com/yourusername/tsagentkit.git
cd tsagentkit
uv sync
```

### Testing

```bash
uv run pytest
```

### Project Structure

```
tsagentkit/
├── src/tsagentkit/
│   ├── contracts/      # Validation and task specs
│   ├── qa/            # Data quality checks
│   ├── series/        # TSDataset and operations
│   ├── router/        # Model selection and fallback
│   ├── models/        # Model adapters
│   │   └── adapters/  # TSFM adapters (Chronos, Moirai, TimesFM)
│   ├── hierarchy/     # Hierarchical reconciliation
│   ├── backtest/      # Rolling window backtesting
│   ├── serving/       # Inference orchestration
│   └── monitoring/    # Drift detection
├── docs/              # Documentation
│   └── README.md      # Documentation index
└── tests/             # Test suite
```

## Roadmap

- **v0.1** ✅: Minimum loop (contracts, QA, series, router, baselines, backtest)
- **v0.2** ✅: Enhanced robustness (monitoring, drift detection, triggers)
- **v1.0** ✅: Ecosystem (TSFM adapters, hierarchical reconciliation)
- **v1.1**: External model registry, custom adapter API
- **v1.2**: Distributed forecasting, streaming support

## Contributing

Contributions welcome! Please read [AGENTS.md](AGENTS.md) for coding guidelines.

## License

Apache-2.0 License - see LICENSE file for details.

## Acknowledgments

- Chronos: Amazon Science
- Moirai: Salesforce AI Research
- TimesFM: Google Research
