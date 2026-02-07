# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tsagentkit` is a Python library that serves as a robust execution engine for external coding agents (LLMs/AI agents) performing time-series forecasting tasks. It provides strict guardrails to enforce proper time-series practices (preventing data leakage, enforcing temporal integrity, etc.).

**Version**: 1.1.0 (Released)

**Key v1.1.0 Change**: TSFM policy now defaults to `mode="required"`, meaning TSFM is required by default. To allow fallback to non-TSFM models, explicitly set `tsfm_policy={"mode": "preferred"}`.

## Python Environment

- Python version: 3.11
- Dependencies: Listed in `pyproject.toml`.
- Always use `uv` to manage dependencies.
  - Add packages to `pyproject.toml` as needed.
  - Use `uv sync` to install dependencies.
  - Use `uv run <command>` to run commands with the correct environment.

## Architecture

The codebase follows this workflow pipeline:

```
validate -> QA -> align_covariates -> series -> route -> backtest -> fit -> predict -> calibrate (optional) -> anomaly (optional) -> package
```

### Module Structure

| Module | Responsibility | Key Output |
|--------|---------------|------------|
| `contracts/` | Data validation, task specifications | `ValidationReport`, `TaskSpec` |
| `qa/` | Data quality checks, leakage detection | `QAReport` |
| `series/` | Time alignment, resampling, sparsity ID | `TSDataset`, `SparsityProfile` |
| `time/` | Frequency inference and future index generation | Frequency utilities |
| `covariates/` | Covariate typing, coverage checks, leakage detection | `CovariateBundle`, `AlignedDataset` |
| `features/` | Feature engineering, covariate alignment | `FeatureMatrix`, signatures |
| `router/` | Model selection, fallback strategies | `Plan`, `FallbackLadder` |
| `models/` | Model adapters and baselines | `ModelArtifact`, `ForecastResult` |
| `backtest/` | Rolling window backtesting | `BacktestReport`, `SegmentMetrics` |
| `eval/` | Metric computation and aggregation | `MetricFrame`, `ScoreSummary` |
| `calibration/` | Interval/quantile calibration | `CalibratorArtifact` |
| `anomaly/` | Anomaly detection using forecast intervals | `AnomalyReport` |
| `serving/` | Batch inference, orchestration | `RunArtifact` |
| `monitoring/` | Drift detection, retrain triggers | `DriftReport`, `CoverageMonitor` |
| `hierarchy/` | Hierarchical forecasting | `HierarchyStructure`, reconciliation |
| `skill/` | Documentation and recipes for AI agents | Recipes, tool maps |

### Layered Dependency Architecture

Dependencies flow from bottom to top (no reverse dependencies):

1. **Layer 1**: `contracts/`, `errors/`
2. **Layer 2**: `series/`, `time/`, `covariates/`, `features/`
3. **Layer 3**: `router/`, `backtest/`, `eval/`, `calibration/`, `anomaly/`
4. **Layer 4**: `models/`
5. **Layer 5**: `serving/` (orchestration layer only)

### Key Design Principles

1. **TSFM-first Strategy**: Time-Series Foundation Models are the primary choice, with automatic fallback to simpler models on failure (when `tsfm_policy.mode != "required"`).

2. **Fallback Ladder**: TSFM -> Lightweight (optional) -> Tree/Baseline -> Naive

3. **Strict Guardrails**:
   - `E_SPLIT_RANDOM_FORBIDDEN`: Random train/test splits are banned
   - `E_COVARIATE_LEAKAGE`: Future leakage detection
   - `E_TSFM_REQUIRED_UNAVAILABLE`: Raised when TSFM is required but unavailable
   - Temporal integrity enforced throughout

4. **Provenance**: Full traceability with signatures for data, features, model config, and plan

## Build, Test, and Development Commands

### Running Tests

```bash
# Run all tests
uv run python -m pytest

# Run tests with coverage
uv run python -m pytest --cov=src/tsagentkit --cov-report=term-missing

# Run specific test file
uv run python -m pytest tests/contracts/test_task_spec.py -v

# Run tests for a specific module
uv run python -m pytest tests/backtest/ -v

# Run TSFM policy matrix checks
uv run python -m pytest tests/ci/test_tsfm_policy_matrix.py -v

# Run real TSFM smoke tests (requires model dependencies)
TSFM_RUN_REAL=1 uv run python -m pytest tests/ci/test_real_tsfm_smoke_gate.py -v
```

### Type Checking

```bash
# Run mypy type checker
uv run mypy src/tsagentkit
```

### Code Formatting and Linting

```bash
# Format code with ruff
uv run ruff format src/

# Check code with ruff
uv run ruff check src/

# Fix auto-fixable issues
uv run ruff check src/ --fix

# Run import linter
uv run lint-imports
```

### Running Examples

```bash
# Run a quick forecast example
uv run python -c "
import pandas as pd
from tsagentkit import TaskSpec, run_forecast

df = pd.DataFrame({
    'unique_id': ['A'] * 30,
    'ds': pd.date_range('2024-01-01', periods=30),
    'y': range(30)
})
result = run_forecast(df, TaskSpec(h=7, freq='D'))
print(result.summary())
"
```

## Coding Conventions

- **Language**: Python
- **Indentation**: 4 spaces; no tabs
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Types**: Add type hints for public APIs and key data structures

## Testing Guidelines

- Place tests in a top-level `tests/` directory that mirrors the package structure (e.g., `tests/contracts/`)
- Prefer deterministic, time-order-safe cases (no random splits; see `E_SPLIT_RANDOM_FORBIDDEN` in the PRD)
- Name tests descriptively (e.g., `test_router_fallback_ladder`)

## Key Documentation

- `docs/PRD.md`: Technical requirements and architecture document
- `docs/ARCHITECTURE.md`: Detailed architecture, layering, and data contracts
- `docs/API_STABILITY.md`: API stability guarantees and compatibility policy
- `docs/RELEASE_V1_1.md`: v1.1 release notes and migration guide
- `AGENTS.md`: Repository guidelines for AI agents
- `skill/README.md`: Agent documentation with module guide
- `skill/tool_map.md`: Complete API reference
- `skill/recipes.md`: Runnable end-to-end examples

## Version Roadmap

- **v0.1** ✅: Minimum loop (contracts, qa, series, basic router, baseline models, rolling backtest)
- **v0.2** ✅: Enhanced robustness (monitoring, advanced router, feature hashing, bucketing)
- **v1.0** ✅: Ecosystem (external adapters, hierarchical reconciliation, structured logging)
- **v1.1** ✅: TSFM-first policy (strict TSFM requirements, real adapter smoke tests, hardened CI gates)

## Quick Reference

### Common Imports

```python
from tsagentkit import (
    TaskSpec,               # Define forecasting tasks
    validate_contract,      # Validate input data
    run_qa,                 # Quality assurance
    align_covariates,       # Align covariates with target
    build_dataset,          # Build TSDataset
    TSDataset,              # Time series dataset
    make_plan,              # Create execution plan
    Plan,                   # Execution plan
    rolling_backtest,       # Backtest with rolling windows
    BacktestReport,         # Backtest results
    evaluate_forecasts,     # Metric computation
    MetricFrame,            # Metrics container
    fit_calibrator,         # Fit prediction calibration
    apply_calibrator,       # Apply calibration
    detect_anomalies,       # Anomaly detection
    package_run,            # Package run results
    run_forecast,           # Main entry point
    # Errors
    TSAgentKitError,
    ESplitRandomForbidden,
    ECovariateLeakage,
    ETSFMRequiredUnavailable,
)
```

### TSFM Policy (v1.1.0)

```python
from tsagentkit import TaskSpec

# Default: TSFM required - fails fast if no TSFM adapter available
spec = TaskSpec(h=7, freq="D")

# Prefer TSFM but allow fallback to non-TSFM models
spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "preferred"})

# Disable TSFM routing entirely
spec = TaskSpec(h=7, freq="D", tsfm_policy={"mode": "disabled"})
```

### Step-Level Pipeline (Assembly-First)

```python
from tsagentkit import (
    TaskSpec, validate_contract, run_qa, build_dataset,
    align_covariates, make_plan, rolling_backtest, package_run
)
from tsagentkit.models import fit, predict
from tsagentkit.eval import evaluate_forecasts

spec = TaskSpec(h=7, freq="D")

# Step-by-step execution
report = validate_contract(df, spec)
report.raise_if_errors()

qa = run_qa(df, spec)
cov_bundle = align_covariates(df, spec)  # If using covariates
dataset = build_dataset(df, spec)

plan, _ = make_plan(dataset, spec)

# Optional backtest
cv_report = rolling_backtest(dataset, spec, plan)

# Fit and predict
model_artifact = fit(dataset, spec, plan)
result = predict(model_artifact, dataset, spec)

# Package results
run_artifact = package_run(dataset, spec, result, backtest_report=cv_report)
```

### Main Entry Point (Convenience Wrapper)

```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec(h=7, freq="D", quantiles=[0.1, 0.5, 0.9])
result = run_forecast(data, spec, mode="standard")

# Access results
forecast_df = result.forecast.df
backtest_metrics = result.backtest_report.aggregate_metrics
```

### Model Adapters

Available adapters are registered in `models/adapters/`:

- **Chronos**: Amazon Chronos TSFM
- **TimesFM**: Google TimesFM
- **Moirai**: Salesforce Moirai
- **StatsForecast**: Statistical baselines (Naive, SeasonalNaive, etc.)
- **Sktime**: Optional classical models

```python
from tsagentkit.models import list_adapter_capabilities

# List available adapters
caps = list_adapter_capabilities()
```
