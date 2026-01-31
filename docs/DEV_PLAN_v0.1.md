# tsagentkit v0.1 Development Plan

> **Document Goal**: Technical implementation plan for the v0.1 "Minimum Loop"
> **Target Audience**: Core developers, AI agents contributing code
> **Status**: Draft - Ready for review

---

## 1. Overview

### 1.1 Scope
This plan defines the implementation details for v0.1: the **Minimum Loop**. The goal is to build a functional end-to-end forecasting pipeline with strict guardrails and provenance tracking.

### 1.2 Success Criteria
- [ ] `run_forecast(standard)` produces valid `RunArtifact` with full provenance
- [ ] Random split attempts are rejected with `E_SPLIT_RANDOM_FORBIDDEN`
- [ ] Future covariate leakage is detected and blocked
- [ ] TSFM failure triggers automatic fallback to baseline models
- [ ] All recipes in `skill/recipes.md` run successfully

---

## 2. Module Implementation Plan

### 2.1 Project Structure

```
tsagentkit/
├── pyproject.toml
├── README.md
├── src/
│   └── tsagentkit/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   ├── schema.py          # FR-1: Data validation schemas
│       │   ├── task_spec.py       # FR-2: Task specification
│       │   ├── results.py         # FR-3: Forecast result structures
│       │   └── errors.py          # Error codes and exceptions
│       ├── qa/
│       │   ├── __init__.py
│       │   ├── checks.py          # FR-4: Quality check implementations
│       │   ├── leakage.py         # FR-6: Leakage detection
│       │   ├── repair.py          # FR-5: Repair strategies
│       │   └── report.py          # QAReport dataclass
│       ├── series/
│       │   ├── __init__.py
│       │   ├── dataset.py         # FR-7: TSDataset implementation
│       │   ├── alignment.py       # FR-7: Timezone/resampling
│       │   └── sparsity.py        # FR-8: Sparsity profiling
│       ├── router/
│       │   ├── __init__.py
│       │   ├── plan.py            # Plan dataclass
│       │   ├── router.py          # FR-13: Model selection logic
│       │   └── fallback.py        # FR-15: Fallback ladder
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py            # FR-16: Unified interface
│       │   ├── baseline.py        # FR-17: Baseline implementations
│       │   └── artifact.py        # ModelArtifact
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py          # FR-18: Rolling window engine
│       │   ├── metrics.py         # FR-19: Metrics calculation
│       │   └── report.py          # BacktestReport
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── inference.py       # FR-21: Batch inference
│       │   └── packaging.py       # FR-22: RunArtifact packaging
│       └── skill/
│           ├── __init__.py
│           ├── README.md          # FR-26: Agent documentation
│           └── recipes.md         # FR-27: Runnable examples
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── contracts/
│   ├── qa/
│   ├── series/
│   ├── router/
│   ├── models/
│   ├── backtest/
│   └── test_integration.py
└── docs/
    ├── PRD.md
    └── DEV_PLAN_v0.1.md
```

---

## 3. Phase-by-Phase Implementation

### Phase 1: Foundation (Week 1)
**Goal**: Core data structures and validation

#### 3.1.1 Dependencies (pyproject.toml)
```toml
[project]
name = "tsagentkit"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]
models = [
    "statsforecast>=1.7.0",
    "utilsforecast>=0.1.0",
]
```

#### 3.1.2 Contracts Module (`contracts/`)

**File: `contracts/errors.py`**
- Define all error codes from PRD Section 6.1
- Custom exception classes with structured metadata

```python
class TSAgentKitError(Exception):
    """Base exception with error code and context."""
    error_code: str
    context: dict

class E_SPLIT_RANDOM_FORBIDDEN(TSAgentKitError):
    error_code = "E_SPLIT_RANDOM_FORBIDDEN"

class E_COVARIATE_LEAKAGE(TSAgentKitError):
    error_code = "E_COVARIATE_LEAKAGE"

# ... etc
```

**File: `contracts/schema.py`**
- `validate_contract(data: Any) -> ValidationReport`
- Check required columns: `unique_id` (str), `ds` (datetime), `y` (numeric)
- Check for duplicates on `(unique_id, ds)`
- Infer and validate frequency

**File: `contracts/task_spec.py`**
- `TaskSpec` Pydantic model (FR-2)
- Fields: `horizon`, `freq`, `rolling_step`, `quantiles`, `covariate_policy`
- Must be JSON-serializable and hashable

**File: `contracts/results.py`**
- `ForecastResult` dataclass (FR-3)
- Fields: `unique_id`, `ds`, `yhat`, `quantiles` (optional)
- Provenance metadata structure

#### 3.1.3 QA Module (`qa/`)

**File: `qa/report.py`**
- `QAReport` dataclass
- Tracks: issues found, repairs applied, leakage detected

**File: `qa/checks.py`**
- Missing value detection per series
- Gap detection in time series
- Outlier detection (IQR method)
- Zero-density calculation

**File: `qa/leakage.py`**
- `detect_covariate_leakage(df, ds_col, horizon)` (FR-6)
- Validate that observed covariates don't extend beyond prediction time
- Raise `E_COVARIATE_LEAKAGE` if violation detected

**File: `qa/repair.py`**
- `RepairStrategy` enum: `INTERPOLATE`, `WINSORIZE`, `DROP`
- Apply repairs with logging to provenance

#### Deliverables
- [ ] `contracts/` module with full validation
- [ ] `qa/` module with leakage detection
- [ ] Unit tests for all validation paths
- [ ] 100% coverage on error code paths

---

### Phase 2: Time Series Handling (Week 1-2)
**Goal**: TSDataset and temporal integrity

#### 3.2.1 Series Module (`series/`)

**File: `series/dataset.py`**
- `TSDataset` class - immutable wrapper around DataFrame
- Enforce schema: `[unique_id, ds, y]` + optional covariates
- Metadata: frequency, timezone, series count

**File: `series/alignment.py`**
- `align_timezone(df, target_tz)` - unify timezones
- `resample_series(df, freq, agg_func)` - sum/mean/last aggregation
- Handle irregular frequencies

**File: `series/sparsity.py`**
- `SparsityProfile` dataclass (FR-8)
- Classify series: `regular`, `intermittent`, `cold_start`, `sparse`
- Metrics: observation ratio, gap statistics

**Key Design Decisions:**
- Use `pandas.DatetimeIndex` with explicit frequency
- Store sparsity classification for router bucketing
- All operations return new instances (immutable)

#### Deliverables
- [ ] `TSDataset` with immutable semantics
- [ ] Sparsity profiling with classification
- [ ] Resampling and alignment utilities
- [ ] Tests for edge cases (DST transitions, missing freq)

---

### Phase 3: Routing and Models (Week 2)
**Goal**: Model selection and baseline implementations

#### 3.3.1 Router Module (`router/`)

**File: `router/plan.py`**
- `Plan` dataclass - the execution plan
- Fields: `primary_model`, `fallback_chain`, `config`, `signature`

**File: `router/router.py`**
- `make_plan(dataset: TSDataset, task_spec: TaskSpec, qa: QAReport) -> Plan`
- Simple rules for v0.1:
  - Regular series -> TSFM (if available)
  - Intermittent -> Seasonal Naive
  - Cold start -> Simple baseline
- Compute plan signature (hash)

**File: `router/fallback.py`**
- `FallbackLadder` class (FR-15)
- Chain: TSFM -> Lightweight -> Tree/Baseline -> Naive
- `execute_with_fallback(fit_func, predict_func, ladder)` pattern

#### 3.3.2 Models Module (`models/`)

**File: `models/base.py`**
- Abstract base class `BaseModel`:
  ```python
  class BaseModel(ABC):
      @abstractmethod
      def fit(self, dataset: TSDataset, plan: Plan) -> ModelArtifact: ...

      @abstractmethod
      def predict(self, dataset: TSDataset, artifact: ModelArtifact,
                  horizon: int) -> ForecastResult: ...
  ```

**File: `models/baseline.py`** (FR-17)
Implement using `statsforecast`:
- `SeasonalNaive` - Seasonal naive baseline
- `SimpleExponentialSmoothing` - ETS variant
- `HistoricAverage` - Moving average baseline

**File: `models/artifact.py`**
- `ModelArtifact` dataclass
- Stores: fitted model, config signature, fit timestamp

**Integration Strategy:**
- Use `statsforecast` for proven baseline implementations
- Wrap in unified interface for consistency
- Store native model in artifact for serialization

#### Deliverables
- [ ] `Plan` generation with bucketing
- [ ] `FallbackLadder` implementation
- [ ] 3 baseline models via statsforecast
- [ ] Unified model interface

---

### Phase 4: Backtesting (Week 3)
**Goal**: Rolling window validation without random splits

#### 3.4.1 Backtest Module (`backtest/`)

**File: `backtest/engine.py`**
- `rolling_backtest(dataset, spec, plan, n_windows)` (FR-18)
- Support `expanding` and `sliding` window strategies
- **CRITICAL**: Enforce temporal ordering, ban random splits
- Raise `E_SPLIT_RANDOM_FORBIDDEN` if shuffling detected

```python
def rolling_backtest(
    dataset: TSDataset,
    spec: TaskSpec,
    plan: Plan,
    n_windows: int = 5,
    window_strategy: Literal["expanding", "sliding"] = "expanding"
) -> BacktestReport:
    # Implementation must:
    # 1. Sort by unique_id, ds
    # 2. Create temporal splits (no randomness)
    # 3. Fit on train, predict on test for each window
    # 4. Collect metrics per window and series
```

**File: `backtest/metrics.py`** (FR-19)
- `wape(y_true, y_pred)` - Weighted Absolute Percentage Error
- `smape(y_true, y_pred)` - Symmetric MAPE
- `mase(y_true, y_pred, y_train)` - Mean Absolute Scaled Error
- `pinball_loss(y_true, y_quantile, tau)` - For quantile forecasts

**File: `backtest/report.py`**
- `BacktestReport` dataclass
- Aggregate metrics, per-series metrics, per-window metrics
- Error distribution by segment

#### Deliverables
- [ ] Rolling window engine (expanding + sliding)
- [ ] Guardrail: `E_SPLIT_RANDOM_FORBIDDEN` on any random split attempt
- [ ] All metrics from PRD implemented
- [ ] Structured backtest report

---

### Phase 5: Serving and Orchestration (Week 3-4)
**Goal**: End-to-end pipeline and packaging

#### 3.5.1 Serving Module (`serving/`)

**File: `serving/inference.py`**
- `predict(dataset, artifact, spec) -> ForecastResult` (FR-21)
- Batch inference with sorted, reproducible output
- Validate input schema matches training

**File: `serving/packaging.py`**
- `RunArtifact` dataclass (FR-22)
- Bundle: forecast, plan, metrics, qa report, provenance
- Serialization: JSON metadata + parquet for data

**File: `serving/provenance.py`**
- `Provenance` dataclass
- Signatures for: data, features, model config, plan
- Run ID generation (UUID)
- Audit trail logging

#### 3.5.2 Main Orchestrator

**File: `tsagentkit/__init__.py` or `pipeline.py`**
- `run_forecast(data, task_spec, mode)` - Unified entry point

```python
def run_forecast(
    data: Any,
    task_spec: TaskSpec,
    mode: Literal['quick', 'standard', 'strict'] = 'standard'
) -> RunArtifact:
    """
    Main entry point for forecasting pipeline.

    Pipeline:
    1. validate -> ValidationReport
    2. run_qa -> QAReport (with leakage detection)
    3. build_dataset -> TSDataset
    4. make_plan -> Plan (with fallback ladder)
    5. rolling_backtest -> BacktestReport
    6. fit -> ModelArtifact
    7. predict -> ForecastResult
    8. package -> RunArtifact
    """
```

Mode behaviors:
- `quick`: Skip backtest, fit on all data
- `standard`: Full pipeline with backtest
- `strict`: Fail on any QA issue (no auto-repair)

#### Deliverables
- [ ] `run_forecast` unified entry point
- [ ] `RunArtifact` with full provenance
- [ ] All three modes implemented
- [ ] Serialization/deserialization

---

### Phase 6: Skills and Documentation (Week 4)
**Goal**: Agent-facing documentation and recipes

#### 3.6.1 Skill Module (`skill/`)

**File: `skill/README.md`**
- "What/When/Inputs/Workflow" format for each module
- Tool map: which function to use for what task
- Guardrail summary: what not to do

**File: `skill/recipes.md`**
Two complete examples:
1. **Retail Daily**: Multiple series, seasonal patterns, covariates
2. **Industrial Hourly**: High frequency, irregular gaps, intermittent demand

Each recipe includes:
- Data preparation
- Task specification
- Running the pipeline
- Interpreting results
- Common pitfalls

#### 3.6.2 Tests

**Structure**: Mirror `src/` structure
- `tests/contracts/test_*.py`
- `tests/qa/test_*.py`
- ... etc

**Integration Tests:**
- `test_integration.py`: End-to-end on synthetic data
- Test all acceptance criteria (A1-A5)

#### Deliverables
- [ ] Complete skill documentation
- [ ] 2 runnable recipes
- [ ] Integration tests passing
- [ ] All acceptance criteria verified

---

## 4. Key Technical Decisions

### 4.1 External Dependencies

| Purpose | Package | Rationale |
|---------|---------|-----------|
| Data handling | `pandas>=2.0.0` | Industry standard, time-series support |
| Validation | `pydantic>=2.0.0` | Type-safe config, JSON serialization |
| Baseline models | `statsforecast` | Fast, robust implementations |
| Utilities | `utilsforecast` | Common TS utilities from Nixtla |

**Not using (v0.1):**
- `sktime`: Complex, heavy dependency
- `tsfresh`: Feature engineering deferred to v0.2
- Custom ML frameworks: Keep v0.1 focused

### 4.2 Guardrail Implementation

| Guardrail | Implementation | Error Code |
|-----------|---------------|------------|
| No random splits | Check for `random_state` or shuffling in split operations | `E_SPLIT_RANDOM_FORBIDDEN` |
| No leakage | Validate covariate timestamps <= prediction time | `E_COVARIATE_LEAKAGE` |
| Temporal ordering | Assert `ds` is sorted within each `unique_id` | `E_CONTRACT_UNSORTED` |
| Schema validation | Pydantic models for all inputs | `E_CONTRACT_MISSING_COLUMN` |

### 4.3 Provenance Strategy

Every `RunArtifact` contains:
```python
provenance = {
    "run_id": "uuid",
    "timestamp": "iso8601",
    "data_signature": "hash of input data",
    "task_signature": "hash of task_spec",
    "plan_signature": "hash of plan",
    "model_signature": "hash of model config",
    "qa_repairs": [list of repairs applied],
    "fallbacks_triggered": [list of fallback events],
}
```

---

## 5. Testing Strategy

### 5.1 Unit Tests
- Every module has corresponding test file
- Mock external dependencies (statsforecast)
- Test all error code paths

### 5.2 Integration Tests
- End-to-end on synthetic datasets
- Test guardrail violations
- Test fallback ladder exhaustion

### 5.3 Acceptance Test Cases

| ID | Test | Expected Result |
|----|------|-----------------|
| A1 | Run full pipeline on valid data | Valid `RunArtifact` with provenance |
| A2 | Attempt random train_test_split | `E_SPLIT_RANDOM_FORBIDDEN` raised |
| A3 | Use future covariate in training | `E_COVARIATE_LEAKAGE` raised |
| A4 | Force TSFM failure | Fallback to baseline, success |
| A5 | Execute both recipes | Success, no errors |

### 5.4 Test Data
Create `tests/fixtures/`:
- `synthetic_daily.csv` - 10 series, 2 years, daily
- `intermittent_series.csv` - Sparse demand patterns
- `with_covariates.csv` - Including known/observed covariates

---

## 6. Milestones and Timeline

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | Foundation | contracts/, qa/, tests |
| 1-2 | Series | TSDataset, sparsity, alignment |
| 2 | Routing/Models | Plan, fallback, 3 baselines |
| 3 | Backtest | Rolling engine, metrics, guardrails |
| 3-4 | Serving | run_forecast(), RunArtifact |
| 4 | Polish | skill/, recipes, integration tests |

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| statsforecast API changes | Medium | Pin version, wrap in our interface |
| Pandas frequency edge cases | Medium | Comprehensive test fixtures |
| Performance on large datasets | Low | v0.1 focuses on correctness over scale |
| Agent misuse of API | Medium | Clear skill docs, strict guardrails |

---

## 8. Post-v0.1 Roadmap

Items deferred to v0.2:
- Feature engineering module (`features/`)
- Monitoring and drift detection (`monitoring/`)
- Advanced router bucketing
- TSFM integration (actual foundation models)
- Hierarchical reconciliation

---

## Appendix A: API Surface (v0.1)

### Public Functions
```python
# contracts
from tsagentkit.contracts import TaskSpec, validate_contract

# qa
from tsagentkit.qa import run_qa, QAReport

# series
from tsagentkit.series import TSDataset, build_dataset

# router
from tsagentkit.router import make_plan, Plan

# backtest
from tsagentkit.backtest import rolling_backtest, BacktestReport

# models
from tsagentkit.models import fit, predict, ModelArtifact

# serving
from tsagentkit.serving import package_run, run_forecast, RunArtifact
```

### Type Hierarchy
```python
TaskSpec -> Plan -> ModelArtifact -> RunArtifact
                ↓
         BacktestReport
                ↓
          ForecastResult
```

---

*Document Version: 0.1.0*
*Last Updated: 2026-01-31*
*Author: Development Team*
