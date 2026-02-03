# Stable API & Compatibility Contract

This document defines the **stable public APIs** for `tsagentkit` and the compatibility rules that must be preserved during rebuilds/refactors. Changes to these APIs require explicit migration guidance and/or deprecation windows.

## Stable Public APIs

The following functions/classes are considered stable and **must remain backward compatible**:

### Orchestration
- `tsagentkit.serving.run_forecast()`
  - Semantics: end-to-end pipeline execution with QA, routing, backtest (when enabled), fit/predict, optional calibration/anomaly/monitoring, and packaging.
  - Return: `RunArtifact` containing at minimum `forecast`, `plan`, `task_spec`, `provenance`, and metadata.

### Contracts & Specs
- `tsagentkit.contracts.TaskSpec`
  - Backward-compatible field names (e.g., `h` vs `horizon`) must continue to work.
- `tsagentkit.contracts.PanelContract`
  - Must continue to support `unique_id`, `ds`, `y` (default columns) and the `aggregation` policy.
- `tsagentkit.contracts.ForecastContract`
  - `model`, `yhat`, quantiles/intervals must remain consistent.

### Validation
- `tsagentkit.contracts.validate_contract()`
  - Must continue to validate schema, apply aggregation rules, and return a `ValidationReport`.

### Backtest
- `tsagentkit.backtest.rolling_backtest()`
  - Must enforce temporal integrity and return a `BacktestReport` compatible with existing consumers.

### Calibration & Anomaly
- `tsagentkit.calibration.fit_calibrator()` / `tsagentkit.calibration.apply_calibrator()`
- `tsagentkit.anomaly.detect_anomalies()`

## Stable Data Contracts (Minimum Columns)

These minimum fields must remain valid in outputs:

- `ForecastResult.df` must include: `unique_id`, `ds`, `model`, `yhat`.
- `CVFrame` (when present) must include: `unique_id`, `ds`, `cutoff`, `model`, `y`, `yhat`.
- `RunArtifact.forecast` must be a `ForecastResult`.

## Compatibility Rules

1. **No breaking changes** to the stable API signature without a deprecation plan.
2. **Field preservation**: existing field names in `TaskSpec`, `ForecastResult`, and `RunArtifact` must not be removed.
3. **Behavioral parity**: refactors must preserve default behavior of `run_forecast()` unless explicitly versioned.
4. **Backwards aliases** (e.g., `horizon` → `h`) must continue to work.
5. **Error codes** in `contracts.errors` must remain stable and documented.

## Deprecation Policy

- Deprecations must be announced in documentation with version tags.
- Provide migration notes and a minimum one-minor-release compatibility window before removal.

## Notes

- This list is intentionally strict to protect agent integrations and automated workflows.
- Additional APIs may be promoted to “stable” by adding them here.
