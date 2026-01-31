# tsagentkit Technical Requirements Document (PRD)

> **Document Goal**: Define the technical baseline, architecture, and verification standards for `tsagentkit`.
> **Target Audience**: LLM/Agents (as developers), System Architects, Data Engineers.

## 1. Project Overview

**tsagentkit** is a Python library designed to be the robust execution engine for **external coding agents** performing time-series forecasting tasks. It provides a strict, production-grade workflow skeleton rather than just a collection of models.

### 1.1 Core Objectives
*   **G1. Enforced Workflow**: Enforce `validate -> QA -> series -> route -> backtest -> fit -> predict -> package` pipeline.
*   **G2. Agent-Friendly**: Provide `skill/` documentation to guide agents to write correct code.
*   **G3. Reliability**: "TSFM-first" strategy with deterministic **Fallback Ladders**.
*   **G4. Provenance**: Full traceability (data/feature/model signatures, plan IDs).
*   **G5. Guardrails**: Strict prevention of data leakage (time-travel) and invalid splits.

### 1.2 Non-Goals
*   **N1**: No ETL / Data Warehouse management.
*   **N2**: No built-in decision optimization (inventory/scheduling).
*   **N3**: No hard dependency on a single model library (framework agnostic).
*   **N4**: Not a self-improving agent (it is a toolkit *for* agents).

---

## 2. System Architecture

### 2.1 Module Structure

| Module | Responsibility | Key Output |
| :--- | :--- | :--- |
| `contracts/` | Data validation, task specifications, error models. | `ValidationReport`, `TaskSpec` |
| `qa/` | Data quality checks, leakage detection, repair strategies. | `QAReport` |
| `series/` | Time alignment, resampling, sparsity identification. | `TSDataset`, `SparsityProfile` |
| `features/` | Feature engineering, covariate alignment, version hashing. | `FeatureMatrix`, `signatures` |
| `router/` | Model selection logic, fallback strategies, bucketing. | `Plan` |
| `models/` | Unified adapter interface, baseline implementations. | `ModelArtifact`, `ForecastResult` |
| `backtest/` | Rolling window backtesting, metrics calculation. | `BacktestReport` |
| `serving/` | Batch inference orchestration, artifact packaging. | `RunArtifact` |
| `monitoring/` | Drift detection, coverage analysis, retrain triggers. | `DriftReport` |
| `skill/` | Documentation & recipes specifically for AI Agents. | `README.md`, `recipes.md` |

### 2.2 Agent Interaction Flow
1.  **Read**: Agent reads `skill/README.md` & `skill/tool_map.md`.
2.  **Code**: Agent generates script using `tsagentkit` modules.
3.  **Execute**: Script runs with mandatory **Guardrails**.
4.  **Fail/Succeed**:
    *   *Violation*: Returns strict Error Code (e.g., `E_SPLIT_RANDOM_FORBIDDEN`).
    *   *Success*: Returns `RunArtifact` with full provenance.

---

## 3. Functional Requirements (FR)

### 3.1 Contracts (`contracts/`)
*   **FR-1 Data Validation**:
    *   **Input**: DataFrame/Arrow.
    *   **Checks**: `unique_id` (str), `ds` (datetime), `y` (num), duplicates, frequency.
    *   **Output**: `ValidationReport`.
*   **FR-2 Task Specification (`TaskSpec`)**:
    *   **Fields**: `horizon`, `freq`, `rolling_step`, `quantiles`, `covariate_policy`.
    *   **Constraint**: Must be JSON-serializable and hashable.
*   **FR-3 Forecast Result**:
    *   **Fields**: `unique_id`, `ds`, `yhat`, `quantiles` (optional).
    *   **Meta**: `provenance` (signatures, plan_id, run_id).

### 3.2 Data QA (`qa/`)
*   **FR-4 Quality Checks**: Missing values, gaps, outliers, zero-density.
*   **FR-5 Repair Strategy**: Configurable (interpolate, winsorize, etc.). All repairs logged to provenance.
*   **FR-6 Leakage Detection**:
    *   **Rule**: Future covariates must be known at prediction time.
    *   **Action**: Reject or drop invalid covariates.

### 3.3 Series (`series/`)
*   **FR-7 Alignment**: Timezone unification, resampling (sum/mean/last).
*   **FR-8 Sparsity Profile**: Identify intermittent/cold-start series for the Router.
*   **FR-9 Hierarchy (Optional)**: Support reconciliation inputs.

### 3.4 Features (`features/`)
*   **FR-10 Feature Factory**: Lags, rolling stats, calendar features (Point-in-Time safe).
*   **FR-11 Covariate Policy**: Strict separation of `known` vs `observed` inputs.
*   **FR-12 Versioning**: Compute `feature_config_hash` for traceability.

### 3.5 Router (`router/`)
*   **FR-13 TSFM-first Routing**: Map `TSDataset` + `SparsityProfile` -> `Plan`.
*   **FR-14 Bucketing**: Strategy for Head vs. Tail / Short vs. Long history.
*   **FR-15 Fallback Ladder**:
    *   **Chain**: TSFM -> Lightweight (opt) -> Tree/Baseline -> Naive.
    *   **Requirement**: Automatic degradation on failure.

### 3.6 Models (`models/`)
*   **FR-16 Interface**:
    *   `fit(dataset, plan) -> ModelArtifact`
    *   `predict(dataset, artifact, horizon) -> ForecastResult`
*   **FR-17 Built-in Baselines**: Seasonal Naive, Moving Average, ETS (at least one required).

### 3.7 Backtest (`backtest/`)
*   **FR-18 Rolling Engine**: Expanding/Sliding window. **NO Random Splits**.
*   **FR-19 Metrics**: WAPE, SMAPE, MASE, Pinball Loss (if quantiles).
*   **FR-20 Diagnostics**: Structured error reports by segment/time.

### 3.8 Serving (`serving/`)
*   **FR-21 Batch Inference**: Reproducible, sorted output.
*   **FR-22 Artifacts**: Comprehensive bundle (forecast, plan, metrics, qa, provenance).

### 3.9 Monitoring (`monitoring/`)
*   **FR-23 Drift**: PSI/KS test on input distributions.
*   **FR-24 Stability**: Prediction jitter, quantile coverage.
*   **FR-25 Triggers**: Rules for retraining (drift threshold or schedule).

### 3.10 Skills (`skill/`)
*   **FR-26 Agent Docs**: Standardized "What/When/Inputs/Workflow" format.
*   **FR-27 Recipes**: Runnable end-to-end examples (Retail Daily, Industrial Hourly).

---

## 4. Constraints & Guardrails

### 4.1 Temporal Integrity
*   **Strict Sequence**: Training and Backtesting must strictly follow time order.
*   **Ban Random Split**: Detection of random shuffling triggers `E_SPLIT_RANDOM_FORBIDDEN`.

### 4.2 Point-in-Time Correctness
*   **Covariates**: `observed` covariates cannot be used for future horizons.
*   **Features**: Lag creation must not peek into the future.

### 4.3 Provenance & Reproducibility
*   **Run ID**: Unique identifier per execution.
*   **Signatures**: Mandatory hashing of Data, Feature Config, Model Config, and Plan.
*   **Audit Trail**: Every repair, fallback, or drop action must be recorded.

---

## 5. API Specification (Minimal Surface)

```python
# Core Workflow
def validate_contract(data: Any) -> ValidationReport: ...
def run_qa(data: Any, task_spec: TaskSpec) -> QAReport: ...
def build_dataset(data: Any, task_spec: TaskSpec) -> TSDataset: ...
def make_plan(dataset: TSDataset, task_spec: TaskSpec, qa: QAReport) -> Plan: ...
def rolling_backtest(dataset: TSDataset, spec: TaskSpec, plan: Plan) -> BacktestReport: ...
def fit(dataset: TSDataset, plan: Plan) -> ModelArtifact: ...
def predict(dataset: TSDataset, artifact: ModelArtifact, spec: TaskSpec) -> ForecastResult: ...
def package_run(...) -> RunArtifact: ...

# Unified Entry Point (Agent Friendly)
def run_forecast(
    data: Any, 
    task_spec: TaskSpec, 
    mode: Literal['quick', 'standard', 'strict'] = 'standard'
) -> RunArtifact: ...
```

---

## 6. Observability & Error Codes

### 6.1 Standard Error Codes
*   `E_CONTRACT_MISSING_COLUMN`: Input schema violation.
*   `E_DUPLICATE_KEY`: Uniqueness constraint violation (`unique_id` + `ds`).
*   `E_SPLIT_RANDOM_FORBIDDEN`: Illegal splitting strategy detected.
*   `E_COVARIATE_LEAKAGE`: Future leakage detected.
*   `E_MODEL_FIT_FAILED`: Training failure (triggers fallback).
*   `E_FALLBACK_EXHAUSTED`: All models in ladder failed.

### 6.2 Structured Logging
*   Events must include: `step_name`, `duration_ms`, `status`, `error_code` (if any), `artifacts_generated`.

---

## 7. Verification & Acceptance Criteria

### 7.1 Must-Have (Blocking)
*   **A1**: `run_forecast(standard)` produces valid `RunArtifact` with full provenance on valid data.
*   **A2**: Random split attempts are rejected with `E_SPLIT_RANDOM_FORBIDDEN`.
*   **A3**: Future leakage of observed covariates is detected and blocked.
*   **A4**: TSFM failure triggers Fallback Ladder to produce a valid baseline forecast.
*   **A5**: `skill/recipes.md` examples run successfully locally.

### 7.2 Should-Have (Quality)
*   **B1**: Sparse/Cold-start data is automatically handled via robust routing.
*   **B2**: Binary reproducibility given same data snapshot and config (seed controlled).

---

## 8. Version Scope

### v0.1: Minimum Loop
*   Modules: `contracts`, `qa`, `series`, `router` (basic), `baseline models`, `rolling backtest`.
*   Feature: End-to-end flow with Provenance.

### v0.2: Enhanced Robustness
*   Modules: `monitoring`, Advanced `router` (bucketing), Full Feature Hashing.

### v1.0: Ecosystem
*   Modules: External Adapters, Hierarchical Reconciliation.
