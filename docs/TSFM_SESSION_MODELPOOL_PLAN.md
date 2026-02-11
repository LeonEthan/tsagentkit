# TSAgentKit Session + ModelPool Refactor Plan

Date: 2026-02-11
Owner: tsagentkit
Status: Draft for implementation

## Goal

Refactor the current function-oriented flow into a session-oriented class design so that:

1. TSFM model weights are preloaded once during session init (up to 3 adapters).
2. `fit` does not load model weights.
3. `predict` does not load model weights.
4. GIFT-Eval batch runs reuse the same in-memory model instances.

This targets repeated Chronos2 loading overhead observed in benchmark runs.

## Non-goals

1. No change to forecast semantics, routing policy, or metric computation.
2. No forced preload for all users by default (must be policy-configurable).
3. No cross-process shared GPU memory design in this phase set.

## Target Architecture

## Runtime Roles

1. `ModelPool` (lifecycle layer)
2. `TSAgentSession` (orchestration layer)

## `ModelPool` Responsibilities

1. Initialize adapter instances.
2. Preload selected adapters (`chronos`, `moirai`, `timesfm`, max 3).
3. Return loaded adapters via `get(adapter_name)`.
4. Track status/metadata (`loaded`, `device`, `model_size`, `load_ms`).
5. Support explicit `close()` / `unload()` for resource release.

## `TSAgentSession` Responsibilities

1. Hold one `ModelPool` for session lifetime.
2. Expose step methods (`validate`, `qa`, `plan`, `fit`, `predict`, `run`).
3. Ensure fit/predict paths consume pool adapters only.
4. Block implicit model loading in fit/predict runtime.

## Phase Plan

## Phase 0: Baseline + Safety Harness

Objective:
Capture baseline behavior and lock regression tests before refactor.

Tasks:
1. Add benchmark timing probe around model loading paths.
2. Add test to assert repeated batch forecasting currently triggers repeated adapter construction/load.
3. Add test fixture for deterministic single-dataset benchmark smoke.

Acceptance:
1. Baseline report includes per-run `load_count`, `load_time_ms_total`, `forecast_time_ms_total`.
2. CI has at least one failing assertion if accidental silent behavior changes are introduced later.

Rollback:
1. Test-only additions, no runtime risk.

## Phase 1: Introduce `ModelPool` (No behavior switch yet)

Objective:
Implement pool class and preload API without changing main execution path.

Tasks:
1. Create `ModelPool` module under serving/runtime area.
2. Define immutable pool config:
   1. `adapters: list[str]`
   2. `model_size_by_adapter`
   3. `device`
   4. `preload: bool`
   5. `max_preload_adapters=3`
3. Implement `preload_all()` with bounded adapter count and deterministic load order.
4. Add metadata stats API (`stats()`).

Acceptance:
1. Unit tests validate:
   1. Preload max-3 constraint.
   2. Repeated `get()` returns same object identity.
   3. `close()` unloads all entries.

Rollback:
1. Keep feature isolated and unused by orchestration.

## Phase 2: Adapter Contract Hardening (No implicit loading)

Objective:
Remove hidden weight loading from adapter `fit`/`predict`.

Tasks:
1. Add strict contract to base adapter:
   1. `fit` requires `is_loaded=True`.
   2. `predict` requires `is_loaded=True`.
2. In each TSFM adapter (Chronos/Moirai/TimesFM), replace implicit:
   1. `if not self.is_loaded: self.load_model()`
   with explicit error (`EModelNotLoaded` style).
3. Update error messages with fix hint:
   1. "Preload adapter in ModelPool or call load_model() during init stage."

Acceptance:
1. Unit tests fail if `fit/predict` tries to auto-load.
2. Existing flows still pass while explicit loading is done by caller.

Rollback:
1. Revert adapter contract hardening commit if migration blockers appear.

## Phase 3: `TSAgentSession` Introduction + Orchestration Wiring

Objective:
Move runtime orchestration from loose function calls to class-backed session.

Tasks:
1. Add `TSAgentSession` with constructor inputs:
   1. `mode`
   2. `task_spec_defaults`
   3. `model_pool`
2. Implement method mapping:
   1. `run()` (full pipeline)
   2. `fit()` (pool-backed adapter resolution)
   3. `predict()` (pool-backed adapter usage)
3. Keep existing `run_forecast()` as compatibility wrapper:
   1. Creates temporary session if caller does not pass one.

Acceptance:
1. Backward-compatible call sites continue to work.
2. Session path produces equivalent forecast outputs for fixed seed data.

Rollback:
1. Revert session wiring commit and keep previous orchestration implementation.

## Phase 4: GIFT-Eval Integration (Primary pain point)

Objective:
Use long-lived session in benchmark predictor to eliminate repeated loads per batch.

Tasks:
1. In `benchmarks/gift_eval/eval/predictor.py`, create one session per predictor instance.
2. Initialize `ModelPool` in predictor init:
   1. preload adapters from policy (default `chronos` first; optional up to 3).
3. Route `_predict_batch()` through session `run()` and shared pool.
4. Add predictor-level teardown to release model memory.

Acceptance:
1. For one dataset run, observed model load count per adapter <= 1 per process.
2. No regression in benchmark output schema and metric computation.
3. End-to-end runtime shows clear reduction vs baseline.

Rollback:
1. Revert benchmark predictor integration commit if benchmark regressions appear.

## Phase 5: Fallback + Multi-adapter Policy Validation

Objective:
Ensure fallback ladders still work under preloaded and partially preloaded scenarios.

Tasks:
1. Test matrix:
   1. preload only chronos
   2. preload chronos+moirai
   3. preload all three
2. Validate fallback behavior when first adapter fails at predict time.
3. Validate guardrail behavior when non-preloaded adapter is selected unexpectedly.

Acceptance:
1. Fallback order and selected model remain policy-consistent.
2. Errors are explicit, actionable, and do not silently auto-load.

Rollback:
1. Allow temporary lazy-load fallback behind explicit flag `allow_lazy_load_on_miss`.

## Phase 6: Cleanup + Default Enablement

Objective:
Enable session+pool path by default and remove obsolete duplicate loading logic.

Tasks:
1. Deprecate direct adapter creation in hot paths.
2. Remove dead code related to unused serving cache path duplication.
3. Update docs:
   1. `docs/ARCHITECTURE.md`
   2. benchmark `README.md`
   3. benchmark `RUNBOOK.md`
4. Add operational notes for GPU memory and preload choices.

Acceptance:
1. Default benchmark path uses session+pool.
2. CI green for unit + integration + benchmark smoke.
3. Measured performance improvement documented.

Rollback:
1. One release cycle with compatibility switch retained before hard removal.

## Implementation Order (Recommended)

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6

Rationale:
1. Build observability first.
2. Introduce lifecycle primitive before contract hardening.
3. Wire orchestration after contracts are strict.
4. Integrate benchmark after runtime foundation is stable.

## Risks and Mitigations

1. Risk: OOM when preloading 3 large adapters on one GPU.
Mitigation:
1. Cap preload count to 3 and make list explicit.
2. Support per-adapter size policy (`small/base`) in preload config.

2. Risk: Regression from removing implicit loading.
Mitigation:
1. Add explicit error type and migration tests.
2. Keep temporary compatibility flag for one cycle.

3. Risk: Legacy call sites bypass session and keep old behavior.
Mitigation:
1. Add runtime warning when compatibility wrapper creates ephemeral session.
2. Add telemetry counter for wrapper usage.

## Metrics to Track

1. `tsfm_model_load_count{adapter}`
2. `tsfm_model_load_ms_sum{adapter}`
3. `session_reuse_count`
4. `forecast_batch_ms`
5. `oom_events`

Success thresholds:
1. `tsfm_model_load_count{chronos}` reduced to 1 per benchmark process in normal runs.
2. End-to-end benchmark runtime reduced materially vs Phase 0 baseline.

## Definition of Done

1. GIFT-Eval predictor runs with one long-lived `TSAgentSession`.
2. `fit/predict` paths do not contain implicit model loading.
3. Model preload policy is explicit, bounded, and documented.
4. Benchmarks confirm reduced loading overhead with no metric regressions.
