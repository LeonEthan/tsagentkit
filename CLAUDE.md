# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tsagentkit` is a Python library that serves as a robust execution engine for external coding agents (LLMs/AI agents) performing time-series forecasting tasks. It provides strict guardrails to enforce proper time-series practices (preventing data leakage, enforcing temporal integrity, etc.).

**Version**: 1.0.0 (Released)

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
validate -> QA -> series -> route -> backtest -> fit -> predict -> package
```

### Module Structure

| Module | Responsibility | Key Output |
|--------|---------------|------------|
| `contracts/` | Data validation, task specifications | `ValidationReport`, `TaskSpec` |
| `qa/` | Data quality checks, leakage detection | `QAReport` |
| `series/` | Time alignment, resampling, sparsity ID | `TSDataset`, `SparsityProfile` |
| `features/` | Feature engineering, covariate alignment | `FeatureMatrix`, signatures |
| `router/` | Model selection, fallback strategies | `Plan` |
| `models/` | Model adapters and baselines | `ModelArtifact`, `ForecastResult` |
| `backtest/` | Rolling window backtesting | `BacktestReport`, `SegmentMetrics`, `TemporalMetrics` |
| `serving/` | Batch inference | `RunArtifact` |
| `monitoring/` | Drift detection, retrain triggers | `DriftReport` |
| `skill/` | Documentation and recipes for AI agents | Recipes, tool maps |
| `hierarchy/` | Hierarchical forecasting | `HierarchyStructure`, reconciliation methods |

### Key Design Principles

1. **TSFM-first Strategy**: Time-Series Foundation Models are the primary choice, with automatic fallback to simpler models on failure.

2. **Fallback Ladder**: TSFM -> Lightweight (optional) -> Tree/Baseline -> Naive

3. **Strict Guardrails**:
   - `E_SPLIT_RANDOM_FORBIDDEN`: Random train/test splits are banned
   - `E_COVARIATE_LEAKAGE`: Future leakage detection
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
result = run_forecast(df, TaskSpec(horizon=7, freq='D'))
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
- `AGENTS.md`: Repository guidelines for AI agents
- `skill/README.md`: Agent documentation with module guide
- `skill/tool_map.md`: Complete API reference
- `skill/recipes.md`: Runnable end-to-end examples

## Version Roadmap

- **v0.1** ✅: Minimum loop (contracts, qa, series, basic router, baseline models, rolling backtest)
- **v0.2** ✅: Enhanced robustness (monitoring, advanced router, feature hashing)
- **v1.0** ✅: Ecosystem (external adapters, hierarchical reconciliation, structured logging)

## Quick Reference

### Common Imports

```python
from tsagentkit import (
    TaskSpec,               # Define forecasting tasks
    validate_contract,      # Validate input data
    run_forecast,           # Main entry point
    TSDataset,              # Time series dataset
    Plan,                   # Execution plan
    BacktestReport,         # Backtest results
    # Errors
    ESplitRandomForbidden,
    ECovariateLeakage,
)
```

### Main Entry Point

```python
from tsagentkit import TaskSpec, run_forecast

spec = TaskSpec(horizon=7, freq="D", quantiles=[0.1, 0.5, 0.9])
result = run_forecast(data, spec, mode="standard")

# Access results
forecast_df = result.forecast.df
backtest_metrics = result.backtest_report.aggregate_metrics
```
