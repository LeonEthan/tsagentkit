# Repository Guidelines

## Python Environment

- Python version: 3.11
- Dependencies: Listed in `pyproject.toml`.
- Always use `uv` to manage dependencies. 
  - Add packages to `pyproject.toml` as needed.
  - Use `uv sync` to install dependencies.
  - Use `uv run <command>` to run commands with the correct environment.

## Project Structure & Module Organization
This repository contains implementation and documentation:
- `docs/PRD.md`: Technical requirements and module layout.
- `src/tsagentkit/`: Core Python packages such as `contracts/`, `qa/`, `series/`, `features/`, `router/`, `models/`, `backtest/`, `serving/`, `monitoring/`, and `skill/`.
- `tests/`: Test suite mirroring the package structure (e.g., `tests/contracts/`).

## Build, Test, and Development Commands
- Run unit/integration tests: `uv run pytest`
- Run real TSFM smoke tests (downloads models): `TSFM_RUN_REAL=1 uv run pytest -m tsfm`
- Prefer single-entry commands (e.g., `python -m pytest`) over custom scripts unless needed.

## Coding Style & Naming Conventions
`tsagentkit` is intended to be a Python library (see `docs/PRD.md`). Until tooling is added, follow standard Python conventions:
- Indentation: 4 spaces; no tabs.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Types: add type hints for public APIs and key data structures.

## Testing Guidelines
- Prefer deterministic, time-order-safe cases (no random splits; see `E_SPLIT_RANDOM_FORBIDDEN` in the PRD).
- Name tests descriptively (e.g., `test_router_fallback_ladder`).
- Document test runner commands in this file and in the README.

## Commit & Pull Request Guidelines
Git history is minimal and uses short, plain summaries (e.g., “init”, “Initial commit”), so no formal convention is established. Use clear, imperative subject lines and keep each commit focused.

For PRs:
- Describe the change, scope, and any new modules.
- Link related issues or PRDs when applicable.
- Call out any deviations from the PRD or new guardrail behavior.

## Agent-Specific Instructions
If you add or update agent-facing documentation, place it under `skill/` and keep the “What/When/Inputs/Workflow” format described in `docs/PRD.md`.
