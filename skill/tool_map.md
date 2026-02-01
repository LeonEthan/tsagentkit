# tsagentkit Tool Map

> **Quick Reference**: All public functions and classes organized by module.

## Table of Contents

1. [Top-Level API](#top-level-api)
2. [Contracts](#contracts)
3. [Series](#series)
4. [QA](#qa)
5. [Router](#router)
6. [Models](#models)
7. [Backtest](#backtest)
8. [Serving](#serving)
9. [Monitoring](#monitoring)
10. [Hierarchy](#hierarchy)
11. [Features](#features)

---

## Top-Level API

Convenient imports from `tsagentkit` package:

```python
from tsagentkit import (
    # Core contracts
    TaskSpec, ValidationReport, ForecastResult, ModelArtifact,
    Provenance, RunArtifact, validate_contract,
    # Errors
    TSAgentKitError, ESplitRandomForbidden, ECovariateLeakage,
    # Series
    TSDataset, SparsityProfile, SparsityClass, build_dataset,
    # QA
    run_qa,
    # Router
    Plan, make_plan, FallbackLadder, execute_with_fallback,
    # Router Bucketing
    DataBucketer, BucketConfig, BucketProfile, BucketStatistics, SeriesBucket,
    # Backtest
    BacktestReport, rolling_backtest, wape, smape, mase,
    # Serving
    run_forecast, MonitoringConfig,
)
```

---

## Contracts

### Data Validation

#### `validate_contract(data: Any) -> ValidationReport`
**Purpose**: Validate input data against required schema.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `data` | Any | Yes | DataFrame or convertible |

**Returns**: `ValidationReport` with valid, errors, warnings, stats

**Example**:
```python
report = validate_contract(df)
if not report.valid:
    report.raise_if_errors()
```

---

### Task Specification

#### `TaskSpec(horizon, freq, **kwargs)`
**Purpose**: Define forecasting task parameters.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `horizon` | int | Yes | - | Number of steps to forecast |
| `freq` | str | Yes | - | Pandas frequency string ("D", "H", "M") |
| `rolling_step` | int | No | horizon | Step size for backtest windows |
| `quantiles` | list[float] | No | None | Quantiles to forecast (e.g., [0.1, 0.5, 0.9]) |
| `covariate_policy` | str | No | "ignore" | "ignore", "known", "observed", "auto" |
| `repair_strategy` | dict | No | None | QA repair configuration |
| `season_length` | int | No | inferred | Seasonal period |
| `valid_from` | str | No | None | Validation start (ISO 8601) |
| `valid_until` | str | No | None | Validation end (ISO 8601) |
| `metadata` | dict | No | {} | User-defined metadata |

**Methods**:
- `model_hash() -> str`: Compute hash for provenance
- `to_signature() -> str`: Human-readable signature

**Example**:
```python
spec = TaskSpec(
    horizon=7,
    freq="D",
    quantiles=[0.1, 0.5, 0.9],
    season_length=7,
)
```

---

### Results

#### `ValidationReport`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `valid` | bool | Whether validation passed |
| `errors` | list[dict] | Validation errors |
| `warnings` | list[dict] | Validation warnings |
| `stats` | dict | Data statistics |

**Methods**:
- `has_errors() -> bool`
- `has_warnings() -> bool`
- `raise_if_errors() -> None`: Raise first error if any
- `to_dict() -> dict`

---

#### `ForecastResult`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `df` | pd.DataFrame | Forecast with columns [unique_id, ds, yhat, ...quantiles] |
| `provenance` | Provenance | Full provenance information |
| `model_name` | str | Name of model that produced forecast |
| `horizon` | int | Forecast horizon |

**Methods**:
- `get_quantile_columns() -> list[str]`: Get quantile column names
- `get_series(unique_id) -> pd.DataFrame`: Get forecast for specific series
- `to_dict() -> dict`

---

#### `ModelArtifact`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `model` | Any | Fitted model object |
| `model_name` | str | Name of the model |
| `config` | dict | Model configuration |
| `signature` | str | Hash of model config |
| `fit_timestamp` | str | ISO timestamp of fitting |
| `metadata` | dict | Additional metadata |

---

#### `RunArtifact`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `forecast` | ForecastResult | The forecast result |
| `plan` | dict | Execution plan used |
| `backtest_report` | dict | Backtest results (if performed) |
| `qa_report` | dict | QA report (if available) |
| `model_artifact` | ModelArtifact | The fitted model |
| `provenance` | Provenance | Full provenance |
| `metadata` | dict | Run metadata |

**Methods**:
- `to_dict() -> dict`
- `summary() -> str`: Human-readable summary

---

#### `Provenance`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | Unique run identifier (UUID) |
| `timestamp` | str | ISO 8601 timestamp |
| `data_signature` | str | Hash of input data |
| `task_signature` | str | Hash of task specification |
| `plan_signature` | str | Hash of execution plan |
| `model_signature` | str | Hash of model config |
| `qa_repairs` | list[dict] | Data repairs applied |
| `fallbacks_triggered` | list[dict] | Fallback events |
| `metadata` | dict | Additional metadata |

---

### Errors

#### Error Classes

| Class | Error Code | When Raised |
|-------|------------|-------------|
| `TSAgentKitError` | `E_UNKNOWN` | Base error class |
| `EContractMissingColumn` | `E_CONTRACT_MISSING_COLUMN` | Missing required columns |
| `EContractInvalidType` | `E_CONTRACT_INVALID_TYPE` | Wrong column types |
| `EContractDuplicateKey` | `E_CONTRACT_DUPLICATE_KEY` | Duplicate (unique_id, ds) pairs |
| `EContractUnsorted` | `E_CONTRACT_UNSORTED` | Data not sorted |
| `ESplitRandomForbidden` | `E_SPLIT_RANDOM_FORBIDDEN` | Random splits detected |
| `ECovariateLeakage` | `E_COVARIATE_LEAKAGE` | Future covariate leakage |
| `EModelFitFailed` | `E_MODEL_FIT_FAILED` | Model training failed |
| `EFallbackExhausted` | `E_FALLBACK_EXHAUSTED` | All models failed |
| `EQACriticalIssue` | `E_QA_CRITICAL_ISSUE` | Critical QA issues in strict mode |
| `EBacktestInsufficientData` | `E_BACKTEST_INSUFFICIENT_DATA` | Not enough data for backtest |

---

## Series

### TSDataset

#### `TSDataset.from_dataframe(data, task_spec, validate=True, compute_sparsity=True)`
**Purpose**: Create immutable time series dataset.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `data` | pd.DataFrame | Yes | - | Input data |
| `task_spec` | TaskSpec | Yes | - | Task specification |
| `validate` | bool | No | True | Validate input |
| `compute_sparsity` | bool | No | True | Compute sparsity profile |

**Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `n_series` | int | Number of unique series |
| `n_observations` | int | Total observations |
| `date_range` | tuple | (min_date, max_date) |
| `series_ids` | list[str] | List of unique series IDs |
| `freq` | str | Frequency from task spec |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `get_series(unique_id) -> pd.DataFrame` | DataFrame | Get data for specific series |
| `filter_series(series_ids) -> TSDataset` | TSDataset | Filter to specific series |
| `filter_dates(start, end) -> TSDataset` | TSDataset | Filter by date range |
| `split_train_test(test_size, test_start) -> (TSDataset, TSDataset)` | tuple | Temporal split |
| `to_dict() -> dict` | dict | Serialize to dict |
| `with_hierarchy(hierarchy) -> TSDataset` | TSDataset | Attach hierarchy |
| `is_hierarchical() -> bool` | bool | Check if has hierarchy |

---

#### `build_dataset(data, task_spec, validate=True, compute_sparsity=True) -> TSDataset`
Convenience function for `TSDataset.from_dataframe()`.

---

### SparsityProfile

#### `SparsityProfile`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `series_profiles` | dict | Per-series sparsity info |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `get_classification(unique_id) -> SparsityClass` | SparsityClass | Get class for series |
| `has_intermittent() -> bool` | bool | Check for intermittent series |
| `has_cold_start() -> bool` | bool | Check for cold-start series |
| `to_dict() -> dict` | dict | Serialize |

---

#### `SparsityClass` (Enum)
- `REGULAR`: Normal time series
- `SPARSE`: Sparse observations
- `INTERMITTENT`: Many zeros (intermittent demand)
- `COLD_START`: Very short history

---

## QA

### `run_qa(data, task_spec, mode="standard", **kwargs) -> QAReport`
**Purpose**: Run data quality checks.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `data` | pd.DataFrame | Yes | - | Input data |
| `task_spec` | TaskSpec | Yes | - | Task specification |
| `mode` | str | No | "standard" | "quick", "standard", "strict" |
| `zero_threshold` | float | No | 0.3 | Threshold for zero density |
| `outlier_z` | float | No | 3.0 | Z-score threshold for outliers |
| `apply_repairs` | bool | No | False | Apply automatic repairs |
| `repair_strategy` | dict | No | None | Repair configuration |

**QAReport Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `issues` | list[dict] | Detected issues |
| `repairs` | list[dict] | Repairs applied |
| `leakage_detected` | bool | Whether leakage was found |

**Methods**:
- `has_critical_issues() -> bool`
- `to_dict() -> dict`

---

## Router

### `make_plan(dataset, task_spec, qa=None, **kwargs) -> Plan`
**Purpose**: Create execution plan for forecasting.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `dataset` | TSDataset | Yes | - | Time series dataset |
| `task_spec` | TaskSpec | Yes | - | Task specification |
| `qa` | QAReport | No | None | QA report for considerations |
| `strategy` | str | No | "auto" | "auto", "baseline_only", "tsfm_first" |
| `use_tsfm` | bool | No | True | Whether to use TSFMs |
| `tsfm_preference` | list[str] | No | ["chronos", "moirai", "timesfm"] | TSFM priority |

**Returns**: `Plan` with model selection and fallback chain.

---

### Plan

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `primary_model` | str | Primary model to use |
| `fallback_chain` | list[str] | Ordered fallback models |
| `config` | dict | Model configuration |
| `strategy` | str | Strategy used |
| `signature` | str | Hash of plan |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `get_all_models() -> list[str]` | list | Primary + fallbacks |
| `to_signature() -> str` | str | Human-readable signature |
| `to_dict() -> dict` | dict | Serialize |
| `from_dict(data) -> Plan` | Plan | Deserialize |

---

### FallbackLadder

**Class Constants**:
| Constant | Value | Description |
|----------|-------|-------------|
| `STANDARD_LADDER` | ["SeasonalNaive", "HistoricAverage", "Naive"] | Standard fallback |
| `INTERMITTENT_LADDER` | ["Croston", "Naive"] | Intermittent demand |
| `COLD_START_LADDER` | ["HistoricAverage", "Naive"] | Cold-start series |

**Class Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `get_ladder(is_intermittent, is_cold_start) -> list[str]` | list | Get appropriate ladder |
| `with_primary(primary, fallbacks, ...) -> list[str]` | list | Create full chain |

---

### `execute_with_fallback(fit_func, dataset, plan, on_fallback=None) -> (Any, str)`
**Purpose**: Execute fit function with automatic fallback.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `fit_func` | Callable | Yes | Function: fit_func(model_name, dataset, config) |
| `dataset` | TSDataset | Yes | Dataset to fit on |
| `plan` | Plan | Yes | Execution plan |
| `on_fallback` | Callable | No | Callback: on_fallback(from_model, to_model, error) |

**Returns**: (result, model_name_that_succeeded)

---

### Bucketing (v0.2+)

#### `DataBucketer`
**Purpose**: Bucket series by characteristics for routing.

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `create_bucket_profile(dataset, sparsity) -> BucketProfile` | BucketProfile | Profile dataset |
| `get_model_recommendation(bucket) -> str` | str | Recommended model |

---

#### `BucketProfile`
**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `buckets` | dict | Series in each bucket |
| `statistics` | BucketStatistics | Aggregated stats |

---

## Models

### `fit(dataset, plan, on_fallback=None) -> ModelArtifact`
**Purpose**: Fit model using plan's fallback ladder.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `dataset` | TSDataset | Yes | Dataset to fit on |
| `plan` | Plan | Yes | Execution plan |
| `on_fallback` | Callable | No | Callback on fallback |

---

### `predict(dataset, artifact, spec) -> ForecastResult`
**Purpose**: Generate predictions.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `dataset` | TSDataset | Yes | Dataset with context |
| `artifact` | ModelArtifact | Yes | Fitted model |
| `spec` | TaskSpec | Yes | Task specification |

---

### Baseline Models

Available baseline models:

| Model | Description | Parameters |
|-------|-------------|------------|
| `Naive` | Last value forecast | - |
| `SeasonalNaive` | Seasonal naive | `season_length` |
| `HistoricAverage` | Mean of history | - |
| `Theta` | Theta method | - |
| `WindowAverage` | Moving average | `window_size` |
| `SeasonalWindowAverage` | Seasonal moving average | `season_length`, `window_size` |
| `AutoETS` | Exponential smoothing | `season_length` |
| `Croston` | Intermittent demand | - |

---

### TSFM Adapters

Available TSFM adapters (if packages installed):

| Adapter | Package | Description |
|---------|---------|-------------|
| `chronos` | `chronos-forecasting` | Amazon Chronos |
| `moirai` | `moirai-forecasting` | Salesforce Moirai |
| `timesfm` | `timesfm` | Google TimesFM |

---

## Backtest

### `rolling_backtest(dataset, spec, plan, **kwargs) -> BacktestReport`
**Purpose**: Rolling window backtesting with temporal integrity.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `dataset` | TSDataset | Yes | - | Dataset to backtest |
| `spec` | TaskSpec | Yes | - | Task specification |
| `plan` | Plan | Yes | - | Execution plan |
| `fit_func` | Callable | No | default_fit | Custom fit function |
| `predict_func` | Callable | No | default_predict | Custom predict function |
| `n_windows` | int | No | 5 | Number of windows |
| `window_strategy` | str | No | "expanding" | "expanding" or "sliding" |
| `min_train_size` | int | No | auto | Minimum training size |
| `step_size` | int | No | horizon | Step between windows |
| `reconcile` | bool | No | True | Reconcile hierarchical forecasts |

---

### BacktestReport

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `n_windows` | int | Number of windows completed |
| `strategy` | str | "expanding" or "sliding" |
| `window_results` | list[WindowResult] | Per-window results |
| `aggregate_metrics` | dict | Aggregated metrics |
| `series_metrics` | dict[str, SeriesMetrics] | Per-series metrics |
| `errors` | list[dict] | Any errors encountered |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `get_metric(name) -> float` | float | Get aggregate metric |
| `get_series_metric(series_id, name) -> float` | float | Get per-series metric |
| `get_best_series(metric) -> str` | str | Best performing series |
| `get_worst_series(metric) -> str` | str | Worst performing series |
| `summary() -> str` | str | Human-readable summary |
| `to_dict() -> dict` | dict | Serialize |

---

### Metrics Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `wape(y_true, y_pred) -> float` | (array, array) -> float | Weighted Absolute Percentage Error |
| `smape(y_true, y_pred) -> float` | (array, array) -> float | Symmetric MAPE |
| `mase(y_true, y_pred, y_train, season_length) -> float` | (array, array, array, int) -> float | Mean Absolute Scaled Error |
| `mae(y_true, y_pred) -> float` | (array, array) -> float | Mean Absolute Error |
| `rmse(y_true, y_pred) -> float` | (array, array) -> float | Root Mean Squared Error |
| `pinball_loss(y_true, y_quantile, quantile) -> float` | (array, array, float) -> float | Pinball loss for quantiles |

---

## Serving

### `run_forecast(data, task_spec, mode="standard", **kwargs) -> RunArtifact`
**Purpose**: Complete forecasting pipeline.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `data` | pd.DataFrame | Yes | - | Input data |
| `task_spec` | TaskSpec | Yes | - | Task specification |
| `mode` | str | No | "standard" | "quick", "standard", "strict" |
| `fit_func` | Callable | No | None | Custom fit function |
| `predict_func` | Callable | No | None | Custom predict function |
| `monitoring_config` | MonitoringConfig | No | None | Monitoring config |
| `reference_data` | pd.DataFrame | No | None | For drift detection |
| `repair_strategy` | dict | No | None | QA repair config |
| `hierarchy` | HierarchyStructure | No | None | Hierarchy for reconciliation |

**Pipeline Steps**:
1. Validate
2. QA
3. Drop future rows
4. Build dataset
5. Make plan
6. Backtest (standard/strict mode)
7. Fit model
8. Predict
9. Drift detection (if enabled)
10. Create provenance
11. Package

---

### MonitoringConfig

**Fields**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | bool | No | False | Enable monitoring |
| `drift_method` | str | No | "psi" | "psi" or "ks" |
| `drift_threshold` | float | No | None | Drift threshold |
| `check_stability` | bool | No | False | Check stability |
| `jitter_threshold` | float | No | 0.1 | Jitter warning threshold |

---

### `package_run(forecast, plan, **kwargs) -> RunArtifact`
**Purpose**: Package all run outputs.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `forecast` | ForecastResult | Yes | Forecast result |
| `plan` | Plan | Yes | Execution plan |
| `backtest_report` | BacktestReport | No | Backtest results |
| `qa_report` | QAReport | No | QA report |
| `model_artifact` | ModelArtifact | No | Fitted model |
| `provenance` | Provenance | No | Provenance info |
| `metadata` | dict | No | Additional metadata |

---

## Monitoring

### DriftDetector

#### `DriftDetector(method="psi", threshold=None)`
**Purpose**: Detect data drift between reference and current data.

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `detect(reference_data, current_data) -> DriftReport` | DriftReport | Run drift detection |

---

### StabilityMonitor

#### `StabilityMonitor(jitter_threshold=0.1)`
**Purpose**: Monitor prediction stability.

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `compute_stability(forecasts) -> StabilityReport` | StabilityReport | Compute stability metrics |

---

### TriggerEvaluator

#### `TriggerEvaluator(triggers=None)`
**Purpose**: Evaluate retrain triggers.

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(drift_report, stability_report, schedule) -> TriggerResult` | TriggerResult | Evaluate triggers |
| `should_retrain(...) -> bool` | bool | Quick check |

---

## Hierarchy

### HierarchyStructure

#### `HierarchyStructure(aggregation_graph, bottom_nodes, s_matrix=None)`
**Purpose**: Define hierarchical structure.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `aggregation_graph` | dict | Yes | Parent -> children mapping |
| `bottom_nodes` | list[str] | Yes | Bottom-level series |
| `s_matrix` | np.ndarray | No | Aggregation matrix |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `from_dataframe(df, hierarchy_columns) -> HierarchyStructure` | classmethod | Build from data |
| `get_num_levels() -> int` | int | Number of levels |
| `get_nodes_at_level(level) -> list[str]` | list | Nodes at level |
| `get_level(node) -> int` | int | Level of node |
| `is_coherent(values) -> bool` | bool | Check coherence |

---

### Reconciliation Methods

#### `ReconciliationMethod` (Enum)
- `BOTTOM_UP`: Aggregate from bottom level
- `TOP_DOWN`: Disaggregate from top level
- `MIDDLE_OUT`: Middle-out reconciliation
- `OLS`: Ordinary Least Squares
- `WLS`: Weighted Least Squares
- `MIN_TRACE`: Minimum trace (MinT)

---

### `reconcile_forecasts(base_forecasts, structure, method) -> pd.DataFrame`
**Purpose**: Reconcile forecasts to ensure hierarchy coherence.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `base_forecasts` | pd.DataFrame | Yes | Base forecasts |
| `structure` | HierarchyStructure | Yes | Hierarchy structure |
| `method` | ReconciliationMethod | Yes | Reconciliation method |

---

## Features

### FeatureFactory

#### `FeatureFactory(config)`
**Purpose**: Point-in-time safe feature engineering.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `config` | FeatureConfig | Yes | Feature configuration |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `create_features(dataset, reference_time=None) -> FeatureMatrix` | FeatureMatrix | Create features |
| `get_feature_importance_template() -> dict` | dict | Template for importance |

---

### FeatureConfig

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `lags` | list[int] | Lag periods to create |
| `calendar_features` | list[str] | Calendar features ("dayofweek", "month", etc.) |
| `rolling_windows` | dict | {window: ["mean", "std", ...]} |
| `known_covariates` | list[str] | Known covariate columns |
| `observed_covariates` | list[str] | Observed covariate columns |
| `include_intercept` | bool | Add intercept column |

---

### FeatureMatrix

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `data` | pd.DataFrame | Data with features |
| `config_hash` | str | Hash of feature config |
| `target_col` | str | Target column name |
| `feature_cols` | list[str] | Feature column names |

**Methods**:
| Method | Returns | Description |
|--------|---------|-------------|
| `get_X() -> np.ndarray` | array | Feature matrix |
| `get_y() -> np.ndarray` | array | Target vector |
| `get_signature() -> str` | str | Unique signature |
| `to_dict() -> dict` | dict | Serialize |

---

## Utility Functions

### Quantiles

| Function | Signature | Description |
|----------|-----------|-------------|
| `quantile_col_name(q) -> str` | (float) -> str | Generate quantile column name |
| `parse_quantile_column(col) -> float | None` | (str) -> float | Parse quantile from column name |
| `normalize_quantile_columns(df) -> pd.DataFrame` | (DataFrame) -> DataFrame | Normalize quantile columns |

---

## Version

```python
from tsagentkit import __version__
print(__version__)  # "0.2.0"
```
