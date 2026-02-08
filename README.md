# tsagentkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**What**: Assembly-first execution engine for time-series forecasting. Compose explicit pipelines from modular blocks, or use convenience wrappers.

**When**: Use when you need to forecast time-series data with guardrails against leakage, automatic TSFM routing, and deterministic provenance.

**Inputs**: `pandas.DataFrame` with `unique_id`, `ds`, `y` columns.

---

## Quick Decision Guide

| Your Goal | Recommended API | Lines | Control |
|-----------|----------------|-------|---------|
| Get forecast now | `forecast(df, h=7)` | 2 | Low |
| Production pipeline | `run_forecast(df, spec)` | 3 | Medium |
| Custom assembly | Assembly-First | 10 | High |
| Debug data issues | `diagnose(df)` + `repair()` | 5 | - |

---

## Installation

```bash
# Core only — statistical baselines
pip install tsagentkit

# With TSFM adapters (Chronos, Moirai, TimesFM)
pip install tsagentkit[tsfm]

# With hierarchical reconciliation
pip install tsagentkit[hierarchy]

# With feature engineering
pip install tsagentkit[features]

# Everything
pip install tsagentkit[full]
```

Verify installation:

```bash
python -m tsagentkit doctor
```

---

## Guardrails (Safety First)

tsagentkit enforces strict time-series practices. These **cannot be disabled**:

| Guardrail | Code | Trigger |
|-----------|------|---------|
| No random splits | `E_SPLIT_RANDOM_FORBIDDEN` | Shuffled data |
| Temporal ordering | `E_DS_NOT_MONOTONIC` | Unsorted timestamps |
| Covariate leakage | `E_COVARIATE_LEAKAGE` | Future covariates |
| TSFM required | `E_TSFM_REQUIRED_UNAVAILABLE` | TSFM policy violation |

Handle errors with structured hints:

```python
from tsagentkit import TSAgentKitError, repair

try:
    result = forecast(df, h=7)
except TSAgentKitError as e:
    print(f"{e.code}: {e.hint}")  # Auto-fix suggestion
    df = repair(df, e)            # Apply fix automatically
```

---

## Usage Patterns

### Pattern 1: Zero-Config Forecast (Fastest)

```python
from tsagentkit import forecast

result = forecast(df, h=7)
print(result.forecast.df)
```

**When to use**: Exploration, prototyping, quick answers.

### Pattern 2: Convenience Wrapper (Balanced)

```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec(h=7, freq="D")
result = run_forecast(df, spec, mode="standard")
print(result.forecast.df)
```

**When to use**: Production jobs needing provenance and backtesting.

**TaskSpec presets** (v1.1.1):

```python
# Quick iteration — allows fallback to baselines
spec = TaskSpec.starter(h=7, freq="D")

# Production — requires TSFM, full backtest
spec = TaskSpec.production(h=14, freq="D")
```

### Pattern 3: Assembly-First (Full Control)

```python
from tsagentkit import (
    TaskSpec, validate_contract, run_qa,
    build_dataset, make_plan, fit, predict, package_run,
)

spec = TaskSpec(h=7, freq="D")

# 1) Validate
report = validate_contract(df)
report.raise_if_errors()

# 2) QA check
qa = run_qa(df, spec, mode="standard")

# 3) Build dataset
dataset = build_dataset(df, spec)

# 4) Plan routing
plan, decision = make_plan(dataset, spec)

# 5) Fit & predict
model = fit(dataset, plan)
result = predict(dataset, model, spec)

# 6) Package
artifact = package_run(
    forecast=result,
    plan=plan,
    task_spec=spec.model_dump(),
    qa_report=qa,
    model_artifact=model,
    provenance=result.provenance,
)
```

**When to use**: Custom preprocessing, intermediate inspection, complex pipelines.

### Pattern 4: Diagnose & Repair (Debugging)

```python
from tsagentkit import diagnose, repair

# Dry-run: validate + QA + plan without fitting
report = diagnose(df)
print(report["plan"])  # See what router would select

# Auto-fix common issues
df_fixed = repair(df, error)
```

**When to use**: Data quality issues, understanding routing decisions.

---

## TSFM Policy

Control TSFM (Chronos/Moirai/TimesFM) routing behavior:

```python
from tsagentkit import TaskSpec

# Default: TSFM required, fail if unavailable
spec = TaskSpec(h=7, freq="D")

# Allow fallback to baselines
spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})

# Disable TSFM entirely
spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "disabled"})
```

---

## Hierarchical Forecasting

```python
import numpy as np
from tsagentkit.hierarchy import HierarchyStructure

# Define hierarchy
hierarchy = HierarchyStructure(
    aggregation_graph={"Total": ["A", "B"]},
    bottom_nodes=["A", "B"],
    s_matrix=np.array([[1, 0], [0, 1], [1, 1]]),
)

# Attach and forecast
dataset = dataset.with_hierarchy(hierarchy)
plan, _ = make_plan(dataset, spec)  # Auto-reconciles
```

**Methods**: Bottom-Up, Top-Down, Middle-Out, OLS, WLS, MinT

---

## Artifact Lifecycle

Save and replay forecasts deterministically:

```python
from tsagentkit import (
    save_run_artifact,
    load_run_artifact,
    replay_forecast_from_artifact,
)

# Save
save_run_artifact(artifact, "run.json")

# Load and validate
loaded = load_run_artifact("run.json")

# Replay
result = replay_forecast_from_artifact(loaded)
```

---

## CLI Reference

```bash
# Environment check
python -m tsagentkit doctor

# Version
python -m tsagentkit version

# API schema (JSON)
python -m tsagentkit describe
```

---

## API Quick Reference

```python
from tsagentkit import (
    # High-level (2-3 lines)
    forecast,           # Zero-config forecast
    diagnose,           # Dry-run validation
    repair,             # Auto-fix errors
    describe,           # API discovery

    # Mid-level
    run_forecast,       # Convenience wrapper
    TaskSpec,           # Task configuration

    # Assembly-first (step-by-step)
    validate_contract,  # Input validation
    run_qa,             # Data quality
    build_dataset,      # Create TSDataset
    make_plan,          # Routing plan
    fit,                # Train model
    predict,            # Generate forecast
    package_run,        # Package results

    # Lifecycle
    save_run_artifact,
    load_run_artifact,
    replay_forecast_from_artifact,

    # Hierarchical
    HierarchyStructure,
    reconcile_forecasts,
)
```

---

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
┌─────────────┐
│  Backtest   │── Rolling validation (optional)
└─────────────┘
    │
    ▼
┌─────────────┐
│    Fit      │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Predict   │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Package   │── Provenance, signatures
└─────────────┘
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| `skill/QUICKSTART.md` | 3-minute guide |
| `skill/README.md` | Agent module guide |
| `skill/recipes.md` | End-to-end templates |
| `skill/tool_map.md` | API reference |
| `skill/TROUBLESHOOTING.md` | Error codes & fixes |
| `docs/ARCHITECTURE.md` | System design |
| `docs/API_STABILITY.md` | Compatibility |

---

## Testing

```bash
# Unit tests
uv run pytest

# Real TSFM tests (downloads models)
TSFM_RUN_REAL=1 uv run pytest -m tsfm
```

---

## Roadmap

- **v0.1** ✅: Minimum loop
- **v0.2** ✅: Enhanced robustness
- **v1.0** ✅: TSFM adapters, hierarchical
- **v1.1** ✅: TSFM-required policy
- **v1.1.1** ✅: Error hints, `repair()`, quick APIs, CLI
- **v1.2**: Distributed, streaming

---

## License

Apache-2.0
