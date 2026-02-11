# tsagentkit Troubleshooting Guide

Error code reference with actionable fix hints. Every `TSAgentKitError`
carries an `error_code` and an optional `fix_hint` attribute. Use
`err.to_agent_dict()` to get a structured representation suitable for
programmatic handling.

---

## Quick Triage

```python
from tsagentkit import repair, run_forecast
from tsagentkit.contracts import TSAgentKitError

try:
    result = run_forecast(df, spec)
except TSAgentKitError as e:
    print(e.error_code, e.fix_hint)
    repaired_df, actions = repair(df)
    result = run_forecast(repaired_df, spec)
```

---

## Error Code Reference

### Contract Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_CONTRACT_INVALID` | `EContractInvalid` | Input schema/contract invalid. | Check that your DataFrame matches the expected panel format with `unique_id`, `ds`, `y` columns. |
| `E_CONTRACT_MISSING_COLUMN` | `EContractMissingColumn` | Input data is missing required columns. | Ensure DataFrame contains required columns (`unique_id`, `ds`, `y`). Use `df.rename(columns={...})` to map. |
| `E_CONTRACT_INVALID_TYPE` | `EContractInvalidType` | Column has invalid data type. | Cast columns to expected types: `ds` should be datetime, `y` should be numeric. Use `pd.to_datetime()` and `pd.to_numeric()`. |
| `E_CONTRACT_DUPLICATE_KEY` | `EContractDuplicateKey` | Duplicate (`unique_id`, `ds`) pairs found in data. | Remove duplicates: `df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='last')` |
| `E_FREQ_INFER_FAIL` | `EFreqInferFail` | Frequency cannot be inferred/validated. | Specify freq explicitly in `TaskSpec(freq='D')`, or ensure regular time intervals in data. |
| `E_DS_NOT_MONOTONIC` | `EDSNotMonotonic` | Time index not monotonic per series. | Sort your DataFrame: `df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)` |
| `E_SPLIT_RANDOM_FORBIDDEN` | `ESplitRandomForbidden` | Random train/test splits are strictly forbidden. | Use temporal splits only. Never shuffle or randomize time-series data for train/test splitting. |
| `E_CONTRACT_UNSORTED` | `EContractUnsorted` | Data is not sorted by (`unique_id`, `ds`). Alias for `E_DS_NOT_MONOTONIC`. | Sort your DataFrame: `df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)` |

### QA Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_QA_MIN_HISTORY` | `EQAMinHistory` | Series history too short. | Provide more historical data or lower `backtest.min_train_size` in TaskSpec. |
| `E_QA_REPAIR_PEEKS_FUTURE` | `EQARepairPeeksFuture` | Repair strategy violates point-in-time safety. | Review your QA repair strategy to ensure it does not use future data. Use only backward-looking imputation methods. |
| `E_QA_CRITICAL_ISSUE` | `EQACriticalIssue` | Critical data quality issue detected in strict mode. | Review QA report details. Consider running in `mode='standard'` for diagnostics before strict mode. |
| `E_QA_LEAKAGE_DETECTED` | `EQALeakageDetected` | Data leakage detected in strict mode. | Inspect your feature engineering pipeline for any future information leaking into training data. |

### Covariate Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_COVARIATE_LEAKAGE` | `ECovariateLeakage` | Past/observed covariate leaks into future. | Mark past-only covariates with `role='past'`, or use `align_covariates()` for automatic alignment. |
| `E_COVARIATE_INCOMPLETE_KNOWN` | `ECovariateIncompleteKnown` | Future-known covariate missing in horizon. | Ensure all `role='future_known'` covariates have values for the full forecast horizon. |
| `E_COVARIATE_STATIC_INVALID` | `ECovariateStaticInvalid` | Static covariate has invalid cardinality. | Verify static covariates have exactly one value per `unique_id`. |

### Model Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_MODEL_FIT_FAIL` | `EModelFitFailed` | Model fitting failed. | Check model logs for details. Ensure data has sufficient observations and no NaN values in target column. |
| `E_MODEL_PREDICT_FAIL` | `EModelPredictFailed` | Model prediction failed. | Verify the fitted model artifact is valid and the prediction horizon is within supported bounds. |
| `E_MODEL_LOAD_FAILED` | `EModelLoadFailed` | Model loading failed. | Verify the model file path exists and the serialization format is compatible. |
| `E_ADAPTER_NOT_AVAILABLE` | `EAdapterNotAvailable` | TSFM adapter not available. | Install the required adapter package. Run `python -m tsagentkit doctor` to check adapter status. |
| `E_TSFM_REQUIRED_UNAVAILABLE` | `ETSFMRequiredUnavailable` | TSFM is required by policy but no required adapter is available. | Install TSFM adapters: `pip install tsagentkit[tsfm]`, or set `tsfm_policy={'mode': 'preferred'}` to allow fallback. |
| `E_FALLBACK_EXHAUSTED` | `EFallbackExhausted` | All models in the fallback ladder failed. | Verify data has enough observations (>=2 per series), or relax `router_thresholds`. |
| `E_OOM` | `EOOM` | Out-of-memory during fit/predict. | Reduce batch size, use fewer series, or switch to a lighter model. Consider setting `max_points_per_series_for_tsfm` in RouterThresholds. |

### Task Spec Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_TASK_SPEC_INVALID` | `ETaskSpecInvalid` | Task specification is invalid or incomplete. | Check that `h` (horizon) is a positive integer and `freq` is a valid pandas offset alias. |
| `E_TASK_SPEC_INCOMPATIBLE` | `ETaskSpecIncompatible` | Task spec is incompatible with data. | Verify the configured frequency matches the actual data frequency. Use `diagnose()` to inspect. |

### Artifact Lifecycle Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_ARTIFACT_SCHEMA_INCOMPATIBLE` | `EArtifactSchemaIncompatible` | Serialized artifact schema/type is incompatible. | Ensure the artifact was created with a compatible tsagentkit version. Check `artifact.provenance`. |
| `E_ARTIFACT_LOAD_FAILED` | `EArtifactLoadFailed` | Serialized artifact cannot be loaded safely. | Verify the artifact file is not corrupted and the path is correct. |

### Backtest Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_BACKTEST_FAIL` | `EBacktestFail` | Backtest execution failed. | Check model compatibility with the data. Review backtest configuration in TaskSpec. |
| `E_BACKTEST_INSUFFICIENT_DATA` | `EBacktestInsufficientData` | Not enough data for requested backtest windows. | Reduce `backtest.n_windows` or `backtest.h`, or provide more historical data. |
| `E_BACKTEST_INVALID_WINDOW` | `EBacktestInvalidWindow` | Invalid backtest window configuration. | Ensure `backtest.step > 0`, `backtest.n_windows > 0`, and `backtest.min_train_size > 1`. |

### Calibration & Anomaly Errors

| Code | Class | Description | Fix Hint |
|------|-------|-------------|----------|
| `E_CALIBRATION_FAIL` | `ECalibrationFail` | Calibration failed. | Ensure cross-validation output contains `y` and `yhat` columns with sufficient data. |
| `E_ANOMALY_FAIL` | `EAnomalyFail` | Anomaly detection failed. | Verify the calibrated intervals exist and the anomaly detection method is compatible with your data. |

---

## Programmatic Error Handling

All errors inherit from `TSAgentKitError` and provide structured access:

```python
from tsagentkit.contracts.errors import TSAgentKitError, ERROR_REGISTRY

try:
    result = run_forecast(df, spec)
except TSAgentKitError as e:
    info = e.to_agent_dict()
    # {
    #     "error_code": "E_DS_NOT_MONOTONIC",
    #     "message": "Time index not monotonic for series 'A'",
    #     "fix_hint": "Sort your DataFrame: ...",
    #     "context": {"series": "A"}
    # }
```

### Error Registry Lookup

```python
from tsagentkit.contracts.errors import get_error_class, ERROR_REGISTRY

# Get error class by code
cls = get_error_class("E_DS_NOT_MONOTONIC")
print(cls.fix_hint)  # class-level default hint

# Iterate all registered error codes
for code, cls in ERROR_REGISTRY.items():
    print(f"{code}: {cls.fix_hint or '(no default hint)'}")
```

---

## Related Resources

- **Quickstart**: `skill/QUICKSTART.md` — get running in 3 minutes
- **Recipes**: `skill/recipes.md` — end-to-end templates
- **API Reference**: `skill/tool_map.md` — task-to-API lookup
- **Architecture**: `docs/ARCHITECTURE.md` — system design
