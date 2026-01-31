# Repository Guidelines

## Project Structure & Module Organization
This repository currently contains documentation only:
- `docs/PRD.md`: Technical requirements and planned module layout.
- `LICENSE`: Project license.

Implementation is expected to follow the module map in `docs/PRD.md`, with top-level Python packages such as `contracts/`, `qa/`, `series/`, `features/`, `router/`, `models/`, `backtest/`, `serving/`, `monitoring/`, and `skill/`. When code lands, place source files in those package folders and keep any examples or recipes under `skill/` as described in the PRD.

Tests are not present yet. When added, keep them in a top-level `tests/` directory that mirrors the package structure (e.g., `tests/contracts/`).

## Build, Test, and Development Commands
No build or test commands are defined yet because the repository is documentation-only. Once code exists:
- Document new commands here and in the README.
- Prefer single-entry commands (e.g., `python -m pytest`) over custom scripts unless needed.

## Coding Style & Naming Conventions
`tsagentkit` is intended to be a Python library (see `docs/PRD.md`). Until tooling is added, follow standard Python conventions:
- Indentation: 4 spaces; no tabs.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Types: add type hints for public APIs and key data structures.

## Testing Guidelines
There are no tests yet. When introducing tests:
- Prefer deterministic, time-order-safe cases (no random splits; see `E_SPLIT_RANDOM_FORBIDDEN` in the PRD).
- Name tests descriptively (e.g., `test_router_fallback_ladder`).
- Document the test runner command in this file.

## Commit & Pull Request Guidelines
Git history is minimal and uses short, plain summaries (e.g., “init”, “Initial commit”), so no formal convention is established. Use clear, imperative subject lines and keep each commit focused.

For PRs:
- Describe the change, scope, and any new modules.
- Link related issues or PRDs when applicable.
- Call out any deviations from the PRD or new guardrail behavior.

## Agent-Specific Instructions
If you add or update agent-facing documentation, place it under `skill/` and keep the “What/When/Inputs/Workflow” format described in `docs/PRD.md`.
