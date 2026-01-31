# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tsagentkit` is a Python library that serves as a robust execution engine for external coding agents (LLMs/AI agents) performing time-series forecasting tasks. It provides strict guardrails to enforce proper time-series practices (preventing data leakage, enforcing temporal integrity, etc.).

The project is currently in the documentation-only phase. Implementation has not yet begun.

## Planned Architecture

The codebase will follow this workflow pipeline:

```
validate -> QA -> series -> route -> backtest -> fit -> predict -> package
```

### Module Structure (Planned)

| Module | Responsibility | Key Output |
|--------|---------------|------------|
| `contracts/` | Data validation, task specifications | `ValidationReport`, `TaskSpec` |
| `qa/` | Data quality checks, leakage detection | `QAReport` |
| `series/` | Time alignment, resampling, sparsity ID | `TSDataset`, `SparsityProfile` |
| `features/` | Feature engineering, covariate alignment | `FeatureMatrix`, signatures |
| `router/` | Model selection, fallback strategies | `Plan` |
| `models/` | Model adapters and baselines | `ModelArtifact`, `ForecastResult` |
| `backtest/` | Rolling window backtesting | `BacktestReport` |
| `serving/` | Batch inference | `RunArtifact` |
| `monitoring/` | Drift detection, retrain triggers | `DriftReport` |
| `skill/` | Documentation and recipes for AI agents | Recipes, tool maps |

### Key Design Principles

1. **TSFM-first Strategy**: Time-Series Foundation Models are the primary choice, with automatic fallback to simpler models on failure.

2. **Fallback Ladder**: TSFM -> Lightweight (optional) -> Tree/Baseline -> Naive

3. **Strict Guardrails**:
   - `E_SPLIT_RANDOM_FORBIDDEN`: Random train/test splits are banned
   - `E_COVARIATE_LEAKAGE`: Future leakage detection
   - Temporal integrity enforced throughout

4. **Provenance**: Full traceability with signatures for data, features, model config, and plan

## Build, Test, and Development Commands

No build or test commands are defined yet because the repository is documentation-only.

Once code is added:
- Prefer single-entry commands (e.g., `python -m pytest`) over custom scripts
- Document new commands in this file and the README

## Coding Conventions

- **Language**: Python
- **Indentation**: 4 spaces; no tabs
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Types**: Add type hints for public APIs and key data structures

## Testing Guidelines

When introducing tests:
- Place them in a top-level `tests/` directory that mirrors the package structure (e.g., `tests/contracts/`)
- Prefer deterministic, time-order-safe cases (no random splits; see `E_SPLIT_RANDOM_FORBIDDEN` in the PRD)
- Name tests descriptively (e.g., `test_router_fallback_ladder`)

## Key Documentation

- `docs/PRD.md`: Technical requirements and architecture document
- `AGENTS.md`: Repository guidelines for AI agents

## Version Roadmap

- **v0.1**: Minimum loop (contracts, qa, series, basic router, baseline models, rolling backtest)
- **v0.2**: Enhanced robustness (monitoring, advanced router, feature hashing)
- **v1.0**: Ecosystem (external adapters, hierarchical reconciliation)
