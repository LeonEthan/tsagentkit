# tsagentkit Product Requirements Document (PRD) (v1.x - Legacy)

> **⚠️ DEPRECATED**: This document describes the v1.x requirements with competitive model selection. For v2.0 (current minimalist ensemble version), see [PRD_v2.md](./PRD_v2.md).
>
> **Document Goal**: Define the technical baseline, architecture, and verification standards for `tsagentkit` — a **pure time-series forecasting toolbox** designed to be **called by external LLM/Agents**, but containing **no LLM logic itself**.
>
> **Target Audience**: Coding agents (tool callers), System Architects, Data/ML Engineers
> **Last Updated**: 2026-02-02 (Asia/Tokyo)

---

## 1. Project Overview

**tsagentkit** is a Python library that standardizes **data contracts, covariate handling, backtesting, evaluation, anomaly detection, and run packaging** for time-series forecasting.  
It is intended to be the **execution engine** for external coding agents: agents decide *what* to do; `tsagentkit` guarantees *how* it is done (correctly, reproducibly, and auditable).

### 1.1 Core Objectives

1. **Agent-friendly, deterministic workflow**: one canonical pipeline with explicit artifacts at each step.
2. **Production-grade guardrails**: temporal integrity, point-in-time correctness, leakage prevention, and explicit error codes.
3. **Model-agnostic tooling**: adapters for models, not model selection intelligence.
4. **First-class covariates**: explicit support for `static`, `past/observed`, and `future-known` covariates with strict coverage/leakage checks.
5. **First-class anomaly detection**: anomaly scoring based on forecasts and calibrated uncertainty.
6. **Reproducible, auditable outputs**: standardized run artifacts and provenance fields.

### 1.2 Non-Goals

* Any built-in LLM reasoning, tool selection, natural-language explanation, or autonomous decision making.
* Data acquisition (APIs, scraping), feature discovery via LLM, or human-in-the-loop labeling workflows.
* A full training platform (distributed training, experiment management beyond run artifacts).

---

## 2. System Architecture

### 2.1 Module Structure

| Module | Responsibility | Primary Artifacts |
|---|---|---|
| `contracts/` | Input/output schema validation, canonical column names, dtype rules. | `ValidationReport`, `SchemaSpec` |
| `qa/` | Data quality checks + **PIT-safe** repairs (optional). | `QAReport`, `RepairReport` |
| `time/` | Frequency inference/validation, regular grids, future index building. | `TimeIndex`, `FreqSpec` |
| `covariates/` | Covariate typing (`static/past/future`), alignment, coverage/leakage checks, imputation. | `CovariateSpec`, `CovariateBundle`, `AlignedDataset` |
| `features/` | Deterministic feature blocks (lags/rolling/calendar), version hashing. | `FeatureSpec`, `FeatureMatrix`, `signatures` |
| `router/` | Deterministic bucketing, feature-driven candidate pool assembly (TSFM mandatory + statistical models), and selection policy. | `PlanSpec`, `RouteDecision` |
| `models/` | Model protocol + adapters, fit/predict orchestration hooks. | `ModelArtifact`, `ForecastResult` |
| `backtest/` | Competitive rolling-origin CV; all candidates evaluated per unique_id; winner selection. | `CVFrame`, `BacktestReport` |
| `eval/` | Metrics (point + quantile), summaries, leaderboards. | `MetricFrame`, `ScoreSummary` |
| `calibration/` | Forecast uncertainty calibration (e.g., conformal). | `CalibratorArtifact` |
| `anomaly/` | Anomaly detection using (calibrated) forecast uncertainty. | `AnomalyReport`, `AnomalyFrame` |
| `serving/` | Packaging for inference-time usage; IO normalization. | `RunArtifact`, `InferenceBundle` |
| `monitoring/` | Coverage/drift checks, artifact validation at serving time. | `MonitoringReport` |
| `skill/` | Agent-facing docs/specs for tool usage (no execution). | Markdown specs |

> **Note**: `calibration/` is separated from `anomaly/` so anomaly detection can be **calibrated** using CV residuals; models are not required to produce well-calibrated intervals.

### 2.2 Agent Interaction Flow

The library enforces a strict pipeline:

1. **validate**: validate input contract + schema inference
2. **qa**: quality checks + optional PIT-safe repairs
3. **time/covariates**: infer/validate `freq`, align covariates, build future index
4. **features**: generate deterministic features (optional)
5. **router**: deterministic plan creation (model candidates, hyperparams, covariate usage)
6. **backtest**: rolling-origin evaluation to choose plan variants (rule-based)
7. **fit**: fit chosen plan
8. **predict**: produce forecast (point + intervals/quantiles)
9. **calibrate** (optional but recommended): calibrate intervals/quantiles using backtest residuals
10. **detect anomalies** (optional): detect anomalies from (calibrated) uncertainty and actuals
11. **package**: produce a `RunArtifact` with provenance, configs, and outputs

---

## 3. Functional Requirements (FR)

### 3.1 Contracts (`contracts/`)

**FR-1 Canonical Panel Contract**
* Minimum required columns: `unique_id`, `ds`, `y`
* `ds` must be datetime-like; `unique_id` string-like; `y` numeric
* Duplicates on (`unique_id`, `ds`) are rejected unless explicitly aggregated via `task_spec.aggregation`

**FR-2 Forecast Output Contract**
* Standard output columns (long format):
  * `unique_id`, `ds`, `model`, `yhat`
  * Optional: interval columns via `level=[...]` (`yhat_lo_95`, `yhat_hi_95`, ...)
  * Optional: quantiles via `quantiles=[...]` (`q_10`, `q_50`, ...)
* CV outputs must include `cutoff`

**FR-3 SchemaSpec**
* A `SchemaSpec` object can be produced and serialized for reuse, including:
  * column roles, dtypes, and covariate typing if provided by user/agent

---

### 3.2 Data QA (`qa/`)

**FR-4 Data Quality Checks**
* Missingness, outliers, monotonicity of `ds`, minimum history length checks
* Per-series summaries: length, gaps, sparsity, seasonality hints

**FR-5 PIT-safe Repairs (Optional)**
* Repairs must be **causal** (no peeking into the future relative to any cutoff)
* Supported repair strategies:
  * `ffill` / `bfill` **with explicit directionality** (default: causal `ffill`)
  * `winsorize` (by rolling historical quantiles, left-closed windows)
  * `median_filter` (rolling median, left-closed windows)
* Disallowed by default: global interpolation/smoothing that uses future points
* Repair must emit a `RepairReport` detailing:
  * what changed, where, and which strategy was used

---

### 3.3 Time & Index (`time/`)

**FR-6 Frequency Handling**
* `freq` can be explicit in `TaskSpec` or inferred per-series with a global reconciliation
* `make_regular_grid()` can expand to a regular index with explicit fill policies

**FR-7 Future Index Generation**
* `make_future_index(panel, h, freq)` generates future `ds` per `unique_id`
* Ensures compatibility with covariate coverage validation

---

### 3.4 Covariates (`covariates/`)

**FR-8 Covariate Typing**
Covariates are categorized as:
* `static`: constant per `unique_id`
* `past` (aka observed): only available up to forecast start
* `future`: known for the full horizon (e.g., calendar, planned promotions)

**FR-9 Input Modes**
* **Single-table mode**: covariate columns exist in the panel DataFrame; future rows may have `y=NaN`
* **Bundle mode**: `CovariateBundle` with:
  * `static_x`: (`unique_id`, ...)
  * `past_x`: (`unique_id`, `ds`, ...)
  * `future_x`: (`unique_id`, `ds`, ...) covering *all* future steps
* `align_covariates()` returns `AlignedDataset` with unified views.

**FR-10 Coverage & Leakage Guardrails**
* `future` covariates must cover every (`unique_id`, future `ds`) in the horizon; otherwise raise `E_COVARIATE_INCOMPLETE_KNOWN`
* `past` covariates must be null/absent after the forecast start; otherwise raise `E_COVARIATE_LEAKAGE`
* `static` covariates must have exactly one row per `unique_id`; otherwise raise `E_COVARIATE_STATIC_INVALID`

**FR-11 Covariate Policy**
`covariate_policy` (`ignore|known|observed|auto|spec`) determines how columns are typed:
* `ignore`: ignore all extra columns
* `known`: treat all extra columns as `future`
* `observed`: treat all extra columns as `past`
* `auto`: infer `future` if values are present for the full horizon; else `past`
  * **strict behavior**: if a column is inferred as `future` but has missing values in the future index, raise `E_COVARIATE_INCOMPLETE_KNOWN`
* `spec`: use an explicit `CovariateSpec` (recommended for production)

---

### 3.5 Features (`features/`)

**FR-12 Deterministic Feature Blocks**
* Feature blocks must be composable and deterministic:
  * lags, rolling stats, calendar features, simple transforms
* Feature generation must be PIT-safe relative to each fold cutoff
* Each `FeatureMatrix` must have a version hash based on:
  * code version, parameters, and input schema signatures

---

### 3.6 Router (`router/`)

**FR-13 PlanSpec**
A `PlanSpec` is a deterministic, serializable object describing:
* candidate models (assembled via competitive pool, see FR-13a)
* covariate usage rules (`use_static`, `use_past`, `use_future`)
* feature spec reference (optional)
* training window policy (min history, truncation)
* prediction format (point/interval/quantiles, desired levels)
* fallback policy (see FR-15)

**FR-13a Competitive Candidate Pool Assembly**
The router assembles a competitive candidate pool for backtesting:
1. **TSFM is mandatory**: At least one TSFM adapter must be included in candidates (raises `E_TSFM_REQUIRED_UNAVAILABLE` if none available)
2. **Statistical models via feature analysis**: Based on series characteristics from `RouteDecision.stats`, select appropriate statistical models:
   * Short history → Naive, SeasonalNaive only
   * Intermittent demand → Croston, IMAPA
   * Strong seasonality → SeasonalNaive, STL decomposition models
   * High frequency/irregular → Robust baselines
   * Sparse data → Simple baselines (Naive, mean forecast)
3. The assembled pool is evaluated competitively in backtesting (see FR-20)

**FR-13b Model Selection Policy**
Final model selection is performed per `unique_id` based on backtest performance:
* The model with the best `metric` (default: `MASE` or specified in `TaskSpec.selection_metric`) wins
* Ties broken by: 1) simpler model preference, 2) deterministic ordering
* Selection rationale logged per-series in `BacktestReport.selection_decision`

**FR-14 Bucketing Rules (Deterministic)**
A series is bucketed using transparent thresholds (defaults can be overridden):
* `short_history`: length < `min_train_size`
* `sparse`: missing ratio > `max_missing_ratio`
* `intermittent`: intermittency score > `max_intermittency`
* `seasonal_candidate`: seasonality detected with confidence > `min_seasonality_conf`
* `high_frequency`: frequency in ('H', 'min', 'S')
* `strong_trend`: trend strength > trend threshold (via statistical test)
The router produces a `RouteDecision` containing:
* bucket tags
* computed statistics for feature-driven model selection
* selected PlanSpec template (with assembled candidate pool per FR-13a)
* reasons and deterministic feature analysis summary


**Router Default Thresholds (Recommended Defaults)**  
These defaults are intended to be **safe, conservative**, and **overrideable** via `RouterConfig`. They must be applied deterministically.

| Parameter | Default | Applies to | Meaning | Notes |
|---|---:|---|---|---|
| `min_train_size` | 56 | all | Minimum historical points required for non-baseline plans | If `freq` is daily, ~8 weeks. |
| `max_missing_ratio` | 0.15 | all | Max fraction of missing timestamps after regularization | Computed on a regular grid per-series. |
| `max_intermittency_adi` | 1.32 | intermittent | Average demand interval (ADI) threshold | Common intermittent-demand heuristic. |
| `max_intermittency_cv2` | 0.49 | intermittent | Squared coefficient of variation threshold | Used with ADI in Syntetos–Boylan style classification. |
| `min_seasonality_conf` | 0.70 | seasonal_candidate | Minimum confidence to treat as seasonal | Based on your seasonality detector output. |
| `max_series_count_for_tsfm` | 20000 | routing | Max number of series to route into heavy models | Guardrail for cost/latency. |
| `max_points_per_series_for_tsfm` | 5000 | routing | Cap per-series history used by heavy models | Truncate oldest history deterministically. |


**FR-15 Fallback Ladder**
Fallback is deterministic and driven by error classes:
* If **fit fails** (`E_MODEL_FIT_FAIL`, `E_OOM`): try next model candidate
* If **predict fails** (`E_MODEL_PREDICT_FAIL`): try next model candidate
* If **covariate rules fail**: drop to covariate-free plan (if allowed) else abort
The ladder must end with a baseline (`SeasonalNaive` or `Naive`) unless `task_spec.allow_baseline=False`.

> Router does not learn; it only applies rules.

---

### 3.7 Models (`models/`)

**FR-16 Model Protocol**
Every model adapter implements:
* `fit(train: AlignedDataset, plan: PlanSpec) -> ModelArtifact`
* `predict(context: AlignedDataset, artifact: ModelArtifact, plan: PlanSpec) -> ForecastResult`

**FR-17 Built-in Baselines (Core)**
Core package must include:
* `NaiveLastValue`
* `SeasonalNaive` (requires `season_length` or inferred `freq`/seasonality)

**FR-18 Optional Adapters (Extras)**
Adapters in extras may include classical stats/ML/deep/TSFM models. They must not change core contracts.

---

### 3.8 Backtest (`backtest/`)

**FR-19 Rolling-Origin Cross Validation**
* Uses `RollingOriginSplitter(h, n_windows, step, min_train_size)`
* Produces a `CVFrame` with `cutoff`, `y`, and forecast outputs
* Must be PIT-safe for features and repairs

**FR-20 BacktestReport (Competitive Model Selection)**
Backtesting serves as the competitive arena for model selection:
* All candidate models from `PlanSpec` (per FR-13a) are evaluated across all backtest folds
* Per-series metrics computed for each candidate (`MASE`, `sMAPE`, `RMSE`, or specified metric)
* Winner selection per `unique_id` based on best average metric across folds

Includes:
* per-fold metrics per model
* per-series aggregated metrics per model
* competitive ranking matrix (model × unique_id)
* `selection_decision`: mapping of `unique_id` → winning model with rationale
* errors encountered per model candidate
* decision summary with statistical feature analysis that drove candidate pool assembly

---

### 3.9 Evaluation (`eval/`)

**FR-21 Metrics**
Point metrics: `MAE`, `RMSE`, `sMAPE`, `MASE`  
Quantile metrics: `PinballLoss`, `WQL`

**FR-22 Summaries**
* Aggregations by: `model`, `unique_id`, `cutoff`, and overall
* Score tables are stable-schema for downstream tooling

---

### 3.10 Calibration (`calibration/`)

**FR-23 Interval/Quantile Calibration**
* Calibration uses **OOS residuals** from `CVFrame`
* Minimum supported method: `conformal` (global or per-series)
* Output is a `CalibratorArtifact` that can be applied to `ForecastResult`:
  * widen/narrow intervals or adjust quantiles

---

### 3.11 Anomaly Detection (`anomaly/`)

**FR-24 Anomaly Detector API**
* `detect_anomalies()` consumes:
  * `ForecastResult` (ideally calibrated) + actuals `y` if available
* Outputs:
  * `anomaly` (bool), `anomaly_score` (float), `threshold`, `method`, and supporting columns (`lo/hi` or quantiles)

**FR-25 Supported Methods (Core)**
* `interval_breach`: anomaly if `y` outside `[lo, hi]` at chosen `level`
* `normalized_margin`: score based on distance outside interval, normalized by interval width

**FR-26 Calibration Requirement**
* If forecast intervals/quantiles are present but uncalibrated, `detect_anomalies()` should:
  * either require `calibrator` in strict mode, or
  * emit a warning + proceed in standard mode
This is to control false positive rates.

---

### 3.12 Serving (`serving/`)

**FR-27 Inference Bundle**
Serving input must allow:
* last-known history window
* future-known covariates for horizon
* static covariates

**FR-28 RunArtifact Packaging**
`RunArtifact` must include:
* `task_spec`, `plan_spec`, versions, hashes
* `validation_report`, `qa_report`
* `backtest_report` (if any)
* `model_artifact` metadata
* `forecast_result`
* optional `calibration_artifact`
* optional `anomaly_report`

---

### 3.13 Monitoring (`monitoring/`)

**FR-29 Runtime Checks**
* coverage checks (interval hit rate over time)
* drift indicators (simple stats; no heavy ML)
* alert conditions based on anomaly rate thresholds (optional)

---

### 3.14 Skills (`skill/`)

**FR-30 Agent Guidance Specs**
Provide machine-readable and human-readable docs:
* expected inputs/outputs
* canonical workflows
* error code remediation steps
No execution code here.

---

## 4. Constraints & Guardrails

### 4.1 Temporal Integrity
* No random splits; only time-based splits allowed.
* Backtesting must be rolling-origin with explicit `cutoff`.
* All operations must preserve chronological ordering per `unique_id`.

### 4.2 Point-in-Time Correctness (PIT)
* **Covariates**:
  * `past` cannot exist beyond forecast start
  * `future` must be fully known for horizon
* **Features**:
  * lags/rolling computed using left-closed windows (historical only)
* **Repairs**:
  * all repairs must be causal; no global interpolation by default

### 4.3 Provenance & Reproducibility
Each run must produce:
* `run_id`, `plan_id`
* library version + git commit (if available)
* hashes of:
  * input schema signature
  * feature spec signature
  * plan signature
* deterministic seeds (where applicable)

---

## 5. API Specification (Minimal Surface)

```python
# Contracts & QA
def validate_contract(data) -> ValidationReport: ...
def run_qa(data, task_spec) -> QAReport: ...

# Time/Covariates
def make_future_index(panel, h, freq=None): ...
def align_covariates(panel, task_spec, covariates=None) -> AlignedDataset: ...

# Planning / Backtest / Fit / Predict
def make_plan(dataset: AlignedDataset, task_spec, qa: QAReport) -> PlanSpec: ...
def rolling_backtest(dataset: AlignedDataset, spec: TaskSpec, plan: PlanSpec) -> BacktestReport: ...
def fit(dataset: AlignedDataset, plan: PlanSpec) -> ModelArtifact: ...
def predict(dataset: AlignedDataset, artifact: ModelArtifact, spec: TaskSpec) -> ForecastResult: ...

# Calibration + Anomaly
def fit_calibrator(cv: CVFrame, method="conformal", **kwargs) -> CalibratorArtifact: ...
def apply_calibrator(forecast: ForecastResult, calib: CalibratorArtifact) -> ForecastResult: ...
def detect_anomalies(forecast_with_y, method="interval_breach", **kwargs) -> AnomalyReport: ...

# Unified Entry Point (Agent Friendly)
def run_forecast(
    data,
    task_spec,
    covariates=None,
    mode: str = "standard",  # quick|standard|strict
) -> RunArtifact: ...
```

---

## 6. Observability & Error Codes

### 6.1 Standard Error Codes (Non-exhaustive)

| Code | Meaning |
|---|---|
| `E_CONTRACT_INVALID` | Input schema/contract invalid |
| `E_DS_NOT_MONOTONIC` | Time index not monotonic per series |
| `E_FREQ_INFER_FAIL` | Frequency cannot be inferred/validated |
| `E_QA_MIN_HISTORY` | Series history too short |
| `E_QA_REPAIR_PEEKS_FUTURE` | Repair strategy violates PIT |
| `E_COVARIATE_LEAKAGE` | Past/observed covariate leaks into future |
| `E_COVARIATE_INCOMPLETE_KNOWN` | Future-known covariate missing in horizon |
| `E_COVARIATE_STATIC_INVALID` | Static covariate invalid cardinality |
| `E_MODEL_FIT_FAIL` | Model fitting failed |
| `E_MODEL_PREDICT_FAIL` | Model prediction failed |
| `E_TSFM_REQUIRED_UNAVAILABLE` | TSFM is required for backtesting but no adapter available |
| `E_OOM` | Out-of-memory during fit/predict |
| `E_BACKTEST_FAIL` | Backtest execution failed |
| `E_CALIBRATION_FAIL` | Calibration failed |
| `E_ANOMALY_FAIL` | Anomaly detection failed |

### 6.2 Structured Logging

All major steps must log:
* step name, start/end time, durations
* run_id/plan_id
* counts: series, rows, missingness
* errors with code + remediation hints

---

## 7. Verification & Acceptance Criteria

### 7.1 Must-Have (Blocking)

A1. **Temporal integrity**: CV folds and features are PIT-safe; no future leakage.  
A2. **Covariate enforcement**: leakage of `past` covariates is detected and blocked (`E_COVARIATE_LEAKAGE`).  
A3. **Known coverage**: missing `future` covariates for horizon triggers `E_COVARIATE_INCOMPLETE_KNOWN`.  
A4. **Deterministic router**: same inputs produce same PlanSpec + decision reasons.  
A5. **Baseline fallback**: if all candidates fail, baseline runs unless disallowed.  
A6. **RunArtifact completeness**: contains required provenance fields and outputs.
A7. **TSFM mandatory for backtesting**: TSFM is always included in candidate pool; `E_TSFM_REQUIRED_UNAVAILABLE` raised if unavailable.
A8. **Competitive model selection**: Backtest evaluates all candidates per `unique_id`; winner selected based on `selection_metric`.
A9. **Feature-driven candidate assembly**: Statistical models selected based on series characteristics (seasonality, intermittency, trend, etc.).

### 7.2 Should-Have (Quality)

B1. **Calibration**: conformal calibration achieves near-target coverage on synthetic tests (within tolerance).  
B2. **Anomaly control**: for stationary normal series, false positive rate approximately matches selected level (after calibration).  
B3. **Repair auditability**: all repairs emit diffs + strategy metadata.

---

## Appendix A: Key Data Contracts (Summary)

### A.1 PanelFrame
`unique_id: str`, `ds: datetime`, `y: float`

### A.2 ForecastResult
`df` with columns `unique_id, ds, model, yhat` + optional intervals/quantiles, plus `provenance`, `model_name`, `horizon`

### A.3 CVFrame (Competitive Evaluation)
`unique_id, ds, cutoff, model, y, yhat` + optional intervals/quantiles. Note: All candidate models from `PlanSpec` are evaluated for each `unique_id` during competitive backtesting.

### A.4 CovariateBundle
* `static_x(unique_id, ...)`
* `past_x(unique_id, ds, ...)`
* `future_x(unique_id, ds, ...)`



## Appendix B: Pydantic Model Definitions (Normative Reference)

> These models define the **JSON-serializable configuration and artifact contracts** used by agents and orchestration layers.
> DataFrames themselves are not serialized; instead, artifacts reference DataFrame **contracts** (required columns, roles, and invariants).

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------
# Common
# ---------------------------

class BaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CovariateRole = Literal["static", "past", "future_known"]
AggregationMode = Literal["reject", "sum", "mean", "median", "last"]
MissingPolicy = Literal["error", "ffill", "bfill", "zero", "mean"]
IntervalMode = Literal["level", "quantiles"]
AnomalyMethod = Literal["interval_breach", "conformal_interval", "mad_residual"]
SeasonalityMethod = Literal["acf", "stl", "periodogram"]


# ---------------------------
# Data contracts (column-level)
# ---------------------------

class PanelContract(BaseSpec):
    unique_id_col: str = "unique_id"
    ds_col: str = "ds"
    y_col: str = "y"
    aggregation: AggregationMode = "reject"


class ForecastContract(BaseSpec):
    long_format: bool = True
    model_col: str = "model"
    yhat_col: str = "yhat"
    cutoff_col: str = "cutoff"  # required for CV output
    interval_mode: IntervalMode = "level"
    levels: List[int] = Field(default_factory=lambda: [80, 95])
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])


class CovariateSpec(BaseSpec):
    # Explicit typing strongly preferred for agent safety.
    roles: Dict[str, CovariateRole] = Field(default_factory=dict)
    missing_policy: MissingPolicy = "error"


# ---------------------------
# Task / execution specs
# ---------------------------

class BacktestSpec(BaseSpec):
    h: int = Field(..., gt=0)
    n_windows: int = Field(5, gt=0)
    step: int = Field(1, gt=0)
    min_train_size: int = Field(56, gt=1)
    regularize_grid: bool = True


class TaskSpec(BaseSpec):
    # Forecast horizon
    h: int = Field(..., gt=0)

    # Frequency handling
    freq: Optional[str] = None  # e.g. "D", "H", "W"
    infer_freq: bool = True

    # Contracts
    panel_contract: PanelContract = Field(default_factory=PanelContract)
    forecast_contract: ForecastContract = Field(default_factory=ForecastContract)

    # Covariates
    covariates: Optional[CovariateSpec] = None

    # Backtest defaults (can be overridden by the caller)
    backtest: BacktestSpec = Field(default_factory=lambda: BacktestSpec(h=1))


# ---------------------------
# Router / planning
# ---------------------------

class RouterThresholds(BaseSpec):
    min_train_size: int = Field(56, gt=1)
    max_missing_ratio: float = Field(0.15, ge=0.0, le=1.0)

    # Intermittency classification (heuristic, deterministic)
    max_intermittency_adi: float = Field(1.32, gt=0.0)
    max_intermittency_cv2: float = Field(0.49, ge=0.0)

    # Seasonality
    seasonality_method: SeasonalityMethod = "acf"
    min_seasonality_conf: float = Field(0.70, ge=0.0, le=1.0)

    # Trend detection for feature-driven model selection
    min_trend_strength: float = Field(0.6, ge=0.0, le=1.0)

    # Practical routing guardrails
    max_series_count_for_tsfm: int = Field(20000, gt=0)
    max_points_per_series_for_tsfm: int = Field(5000, gt=0)


class PlanSpec(BaseSpec):
    plan_name: str
    candidate_models: List[str] = Field(..., min_length=1)

    # Covariate usage rules
    use_static: bool = True
    use_past: bool = True
    use_future_known: bool = True

    # Training policy
    min_train_size: int = Field(56, gt=1)
    max_train_size: Optional[int] = None  # if set, truncate oldest points deterministically

    # Output policy
    interval_mode: IntervalMode = "level"
    levels: List[int] = Field(default_factory=lambda: [80, 95])
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Fallback policy
    allow_drop_covariates: bool = True
    allow_baseline: bool = True

    # TSFM policy (mandatory for competitive backtesting)
    tsfm_required: bool = True  # If True, raises E_TSFM_REQUIRED_UNAVAILABLE if no TSFM available

    # Model selection policy
    selection_metric: str = "MASE"  # Metric for competitive selection per unique_id


class RouteDecision(BaseSpec):
    # Series statistics used in routing (computed deterministically)
    stats: Dict[str, Any] = Field(default_factory=dict)

    # Bucket tags
    buckets: List[str] = Field(default_factory=list)

    # Which plan template was selected
    selected_plan: PlanSpec

    # Human-readable deterministic reasons (safe for logs)
    reasons: List[str] = Field(default_factory=list)

    # Feature-driven candidate pool assembly metadata
    candidate_pool_assembly: Dict[str, Any] = Field(default_factory=dict)
    # Contains: tsfm_candidates, statistical_candidates, feature_analysis_summary


class RouterConfig(BaseSpec):
    thresholds: RouterThresholds = Field(default_factory=RouterThresholds)

    # Mapping bucket -> plan template name, resolved by registry
    bucket_to_plan: Dict[str, str] = Field(default_factory=dict)

    # Default plan when no bucket matches
    default_plan: str = "default"


# ---------------------------
# Calibration + anomaly
# ---------------------------

class CalibratorSpec(BaseSpec):
    method: Literal["none", "conformal_interval"] = "conformal_interval"
    level: int = Field(99, ge=50, le=99)
    by: Optional[Literal["unique_id", "global"]] = "unique_id"


class AnomalySpec(BaseSpec):
    method: AnomalyMethod = "conformal_interval"
    level: int = Field(99, ge=50, le=99)
    score: Literal["margin", "normalized_margin", "zscore"] = "normalized_margin"


# ---------------------------
# Provenance artifacts
# ---------------------------

class RunArtifact(BaseSpec):
    run_id: str
    created_at: datetime

    task_spec: TaskSpec
    router_config: Optional[RouterConfig] = None
    route_decision: Optional[RouteDecision] = None

    # Identifiers / hashes for reproducibility (implementation-defined)
    data_signature: Optional[str] = None
    code_signature: Optional[str] = None

    # Output references (implementation-defined; typically file paths or object-store keys)
    outputs: Dict[str, str] = Field(default_factory=dict)

    # Competitive backtest results
    model_selection: Optional[Dict[str, Any]] = None
    # Contains: model_rankings, selection_decision per unique_id, winning_model_per_series
```

**Notes**
- `TaskSpec.backtest.h` is set to `1` by default in the model above only because Pydantic requires a value; in implementation, set `backtest.h = TaskSpec.h` if the caller does not override it.
- `RouterThresholds` uses conservative defaults; adjust per business frequency and SLA.
- **Competitive Backtesting Design**: The system is designed as a competitive arena where TSFM (mandatory) and feature-selected statistical models compete in backtesting. The winner per `unique_id` is selected based on the configured `selection_metric` (default: `MASE`).
