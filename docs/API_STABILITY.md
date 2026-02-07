# Stable API & Compatibility Contract

This document defines the **stable public APIs** for `tsagentkit` and the compatibility rules that must be preserved during rebuilds/refactors. Changes to these APIs require explicit migration guidance and/or deprecation windows.

Per `docs/ADR_001_assembly_first.md`, `tsagentkit` is **assembly-first**:
step-level APIs are primary for coding-agent system construction, while
`run_forecast()` is a convenience wrapper.

## Stable Public APIs

The following functions/classes are considered stable and **must remain backward compatible**:

### Primary Step APIs (Assembly Path)
- `tsagentkit.contracts.validate_contract()`
- `tsagentkit.qa.run_qa()`
- `tsagentkit.covariates.align_covariates()`
- `tsagentkit.series.TSDataset.from_dataframe()`
- `tsagentkit.series.build_dataset()`
- `tsagentkit.router.make_plan()`
- `tsagentkit.backtest.rolling_backtest()`
- `tsagentkit.models.fit()`
- `tsagentkit.models.predict()`
- `tsagentkit.serving.package_run()`
- `tsagentkit.serving.save_run_artifact()`
- `tsagentkit.serving.load_run_artifact()`
- `tsagentkit.serving.validate_run_artifact_for_serving()`
- `tsagentkit.serving.replay_forecast_from_artifact()`

### Orchestration Convenience API
- `tsagentkit.serving.run_forecast()`
  - Semantics: end-to-end pipeline wrapper using stable step APIs.
  - Return: `RunArtifact` containing at minimum `forecast`, `plan`, `task_spec`, `provenance`, and metadata.

### Contracts & Specs
- `tsagentkit.contracts.TaskSpec`
  - Backward-compatible field names (e.g., `h` vs `horizon`) must continue to work.
- `tsagentkit.contracts.TSFMPolicy`
  - `mode` semantics (`preferred`, `required`, `disabled`) must remain stable.
  - Default policy is TSFM-first (`mode="required"`); relaxing to baseline fallback must be explicit.
  - `require_tsfm` and `tsfm_preference` compatibility aliases must continue to map into `TaskSpec.tsfm_policy`.
- `tsagentkit.contracts.PanelContract`
  - Must continue to support `unique_id`, `ds`, `y` (default columns) and the `aggregation` policy.
- `tsagentkit.contracts.ForecastContract`
  - `model`, `yhat`, quantiles/intervals must remain consistent.

### Calibration & Anomaly
- `tsagentkit.calibration.fit_calibrator()` / `tsagentkit.calibration.apply_calibrator()`
- `tsagentkit.anomaly.detect_anomalies()`

## Stable Data Contracts (Minimum Columns)

These minimum fields must remain valid in outputs:

- `ForecastResult.df` must include: `unique_id`, `ds`, `model`, `yhat`.
- `CVFrame` (when present) must include: `unique_id`, `ds`, `cutoff`, `model`, `y`, `yhat`.
- `RunArtifact.forecast` must be a `ForecastResult`.
- `RunArtifact` payload metadata must retain:
  - `artifact_type`
  - `artifact_schema_version`
  - `lifecycle_stage`

## Compatibility Rules

1. **No breaking changes** to the stable API signature without a deprecation plan.
2. **Field preservation**: existing field names in `TaskSpec`, `ForecastResult`, and `RunArtifact` must not be removed.
3. **Behavioral parity**: refactors must preserve composition semantics of step APIs and default behavior of `run_forecast()` unless explicitly versioned.
4. **Backwards aliases** (e.g., `horizon` → `h`) must continue to work.
5. **Error codes** in `contracts.errors` must remain stable and documented.
6. **Assembly-first guarantee**: documentation and official examples must keep step-by-step composition as the primary integration pattern.

## Deprecation Policy

- Deprecations must be announced in documentation with version tags.
- Provide migration notes and a minimum one-minor-release compatibility window before removal.

## Notes

- This list is intentionally strict to protect agent integrations and automated workflows.
- Additional APIs may be promoted to “stable” by adding them here.
