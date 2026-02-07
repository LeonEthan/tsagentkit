# ADR 001: Assembly-First Agent Integration

- Status: Accepted
- Date: 2026-02-07
- Owners: tsagentkit maintainers

## Context

The primary user journey is: a human asks a coding agent to build a production-ready
time-series forecasting system, and the coding agent assembles that system using
`tsagentkit` modules.

In this workflow, agent reliability depends on explicit composable interfaces and
stable step-level contracts. A single one-click orchestration API is useful, but it
cannot be the only or primary integration pattern.

## Decision

`tsagentkit` adopts an assembly-first API posture:

1. Step-level APIs are primary and stable:
   - `validate_contract`
   - `run_qa`
   - `align_covariates`
   - `TSDataset.from_dataframe` / `build_dataset`
   - `make_plan`
   - `rolling_backtest`
   - `models.fit` / `models.predict`
   - `package_run`
2. `run_forecast()` remains stable as a convenience wrapper, not the recommended
   default for agent system construction.
3. Documentation and examples must lead with composable, step-by-step assembly flows.
4. Backward compatibility for existing integrations is preserved; no immediate removal
   of orchestration entry points.

## Consequences

### Positive

- Better fit for coding-agent workflows that require explicit planning and control.
- Easier extension points for custom routing, model adapters, and validation policies.
- Clearer observability and debugging between pipeline stages.

### Tradeoffs

- Slightly more work for simple scripts that only need a quick forecast.
- More API surface to document and test for compatibility.

## Compatibility and Migration

- Existing `run_forecast()` integrations continue to work.
- New documentation and stability guarantees prioritize modular assembly flows.
- Future deprecations (if any) must follow the documented compatibility window and
  migration notes in `docs/API_STABILITY.md`.
