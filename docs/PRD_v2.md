# tsagentkit v2.0 Product Requirements Document (PRD)

> **Document Goal**: Define the minimalist, production-grade forecasting toolbox for AI agents. v2.0 is a deliberate simplification from v1.x's complexity toward an ensemble-first paradigm.
>
> **Key Change**: Replaced competitive model selection with **ensemble forecasting**—all models participate, final prediction is median/mean aggregate.
>
> **Target Audience**: Coding agents, ML engineers, data scientists
> **Last Updated**: 2026-02-15

---

## 1. Project Overview

**tsagentkit** is a strict, minimalist execution engine for time-series forecasting. It enforces temporal integrity and provides guardrails, but deliberately avoids complexity in favor of simplicity and robustness.

### 1.1 Core Philosophy: Ensemble > Selection

**v1.x Approach**: Run competitive backtests, select best model per series, fit winner.
**v2.0 Approach**: Run ALL models, ensemble their predictions via median/mean.

**Why Ensemble Won:**
1. **Simplicity**: No per-series model selection logic, no winner-tracking, no fallback chains
2. **Robustness**: Single model failure doesn't break forecast; outliers are dampened by aggregation
3. **Predictability**: Same models run every time; no "why was this model chosen?" questions
4. **Speed**: Parallel model fitting is easier than sequential competitive evaluation
5. **Code Size**: ~1,800 lines vs ~10,000+ lines in v1.x

### 1.2 Core Objectives

1. **Minimalist API**: Single `forecast(df, h=7)` entry point with sensible defaults
2. **TSFM-First Ensemble**: TSFM models (Chronos, TimesFM, Moirai) + statistical baselines all participate
3. **Production Guardrails**: Temporal integrity, no data leakage, explicit error codes
4. **Deterministic Output**: Same input → same ensemble behavior (modulo model availability)
5. **Graceful Degradation**: Works with only statistical models if TSFM unavailable

### 1.3 Non-Goals

* Competitive model selection per series (eliminated in v2.0)
* Per-series hyperparameter tuning
* Complex feature engineering pipelines
* Heavy monitoring/drift detection infrastructure
* Hierarchical reconciliation (out of scope for minimalist version)
* LLM reasoning or autonomous decision-making

---

## 2. System Architecture

### 2.1 Module Structure (v2.0 Minimalist)

| Module | Responsibility | Key Output | Lines (approx) |
|---|---|---|---|
| `core/` | Config, data structures, errors, results | `ForecastConfig`, `TSDataset`, `RunResult` | 450 |
| `models/` | Model adapters and baselines | `ModelArtifact`, ensemble predictions | 350 |
| `models/adapters/` | TSFM wrappers (Chronos, TimesFM, Moirai) | Adapter instances | 400 |
| `pipeline/` | Pipeline stages and runner | `PipelineStage`, `forecast()` | 550 |
| `router/` | Plan building with ensemble configuration | `Plan`, `ModelCandidate` | 170 |
| `qa/` | Basic QA checks | `QAReport` (simplified) | 70 |

**Total Target**: <2,000 lines (currently ~1,800)

### 2.2 Architecture Principles

1. **Ensemble by Default**: All available models participate; no model selection
2. **Thin Adapters**: Delegate to mature libraries (statsforecast, TSFM packages)
3. **Functional Composition**: Pure functions over class hierarchies
4. **Fail Fast on Principles**: Clear errors for temporal violations, graceful on model failures

### 2.3 Pipeline Flow (v2.0)

```
validate → qa → build_dataset → build_ensemble_plan → fit_all_models → predict_all → median_ensemble → package
```

**Key Changes from v1.x:**
- Removed: `rolling_backtest` for model selection
- Removed: Per-series winner selection
- Added: Parallel model fitting
- Added: Median/mean ensemble aggregation

---

## 3. Functional Requirements

### 3.1 Core Configuration

**FR-1 ForecastConfig** (replaces TaskSpec/PlanSpec/RouterConfig)
```python
@dataclass(frozen=True)
class ForecastConfig:
    h: int                          # Forecast horizon
    freq: str = "D"                 # Frequency
    quantiles: list[float] = [0.1, 0.5, 0.9]

    # Ensemble configuration
    ensemble_method: Literal["median", "mean"] = "median"
    require_all_tsfm: bool = False  # Fail if any TSFM fails?
    min_models_for_ensemble: int = 1

    # TSFM policy
    tsfm_mode: Literal["required", "preferred", "disabled"] = "preferred"

    # Execution
    mode: Literal["quick", "standard", "strict"] = "standard"
    n_backtest_windows: int = 0     # v2.0: disabled (was for competitive selection)
```

**Rationale**: Single config object replaces 3+ Pydantic models. Simpler, no validation overhead.

### 3.2 Data Contracts

**FR-2 Canonical Panel Contract**
- Required columns: `unique_id`, `ds`, `y`
- `ds` datetime-like, `unique_id` string-like, `y` numeric
- No duplicates on (`unique_id`, `ds`)

**FR-3 Forecast Output Contract**
- Columns: `unique_id`, `ds`, `yhat`, `_ensemble_count`
- Optional: quantile columns (`q_10`, `q_50`, `q_90`)
- `_ensemble_count`: number of models that contributed (transparency)

### 3.3 Ensemble System

**FR-4 Model Participation**
All models that successfully fit participate in the ensemble:
1. TSFM models (chronos, timesfm, moirai) - if available
2. Statistical models (SeasonalNaive, HistoricAverage, Naive) - always included

**FR-5 Ensemble Aggregation**
- **Median** (default): Element-wise median across all model predictions
- **Mean**: Element-wise mean across all model predictions
- Missing values from failed models are excluded

**FR-6 Minimum Models Requirement**
- `min_models_for_ensemble` (default: 1) - minimum successful models required
- If fewer models succeed, raise `EModelFailed`

**FR-7 TSFM Policy**
- `required`: At least one TSFM must be available; raise `ETSFMRequired` if none
- `preferred`: Use TSFM if available, statistical-only if not
- `disabled`: Skip TSFM, use statistical models only

### 3.4 Models

**FR-8 Model Protocol**
```python
def fit(dataset: TSDataset) -> ModelArtifact: ...
def predict(dataset: TSDataset, artifact: ModelArtifact, h: int) -> pd.DataFrame: ...
```

**FR-9 Built-in Baselines**
- `Naive`: Last value carry-forward
- `SeasonalNaive`: Seasonal last value (requires `season_length`)
- `HistoricAverage`: Mean of historical values

**FR-10 TSFM Adapters**
- `chronos`: Amazon Chronos2 adapter
- `timesfm`: Google TimesFM 2.5 adapter
- `moirai`: Salesforce Moirai adapter
- Adapters loaded lazily; fail gracefully if packages not installed

### 3.5 QA and Validation

**FR-11 Validation**
- Check required columns present
- Check no nulls in key columns
- Normalize column names to standard (`unique_id`, `ds`, `y`)

**FR-12 QA Checks**
- Series length checks (warn if < `min_train_size`)
- Duplicate detection on (`unique_id`, `ds`)
- Temporal ordering check (monotonic `ds` per series)

**FR-13 PIT-Safe Repairs (Optional)**
- Causal forward-fill only (`ffill`)
- No interpolation that uses future points
- Repairs emit warnings, not separate reports

### 3.6 Error Handling

**FR-14 Core Error Types**
```python
class TSAgentKitError(Exception):
    error_code: str
    message: str
    context: dict
    fix_hint: str

class EContractViolation(TSAgentKitError): ...
class EDataQuality(TSAgentKitError): ...
class EModelFailed(TSAgentKitError): ...
class ETSFMRequired(TSAgentKitError): ...
```

**FR-15 Error Codes**
| Code | Meaning |
|---|---|
| `E_CONTRACT_VIOLATION` | Input schema invalid |
| `E_DATA_QUALITY` | Data quality issues (duplicates, ordering) |
| `E_MODEL_FAILED` | All models failed or min_models not met |
| `E_TSFM_REQUIRED` | TSFM required but no adapters available |

---

## 4. API Specification

### 4.1 Primary API

```python
# Main entry point
def forecast(
    data: pd.DataFrame,
    h: int,
    freq: str = "D",
    **config_kwargs
) -> RunResult: ...

# Configuration-based
def run_pipeline(
    data: pd.DataFrame,
    config: ForecastConfig,
) -> RunResult: ...

# Inspection
def inspect_tsfm_adapters() -> list[str]: ...
```

### 4.2 Configuration Presets

```python
# Quick experimentation
ForecastConfig.quick(h=7, freq="D")
# - ensemble_method="median"
# - tsfm_mode="preferred"
# - mode="quick"

# Standard production
ForecastConfig.standard(h=14, freq="D")
# - ensemble_method="median"
# - tsfm_mode="required"
# - mode="standard"

# Strict/Fail-fast
ForecastConfig.strict(h=14, freq="D")
# - tsfm_mode="required"
# - require_all_tsfm=True
# - mode="strict"
```

### 4.3 Result Structure

```python
@dataclass
class RunResult:
    forecast: ForecastResult      # df with yhat, _ensemble_count
    duration_ms: float
    model_used: str               # "ensemble_median" or "ensemble_mean"
    model_errors: list[dict]      # Which models failed and why

    def to_dataframe(self) -> pd.DataFrame: ...
    def summary(self) -> dict: ...
```

---

## 5. Trade-off Analysis: v1.x vs v2.0

### 5.1 What Was Removed (And Why)

| v1.x Feature | v2.0 Status | Rationale |
|---|---|---|
| Competitive backtesting | **Removed** | Ensemble doesn't need model selection; parallel fitting is faster |
| Per-series model selection | **Removed** | Ensemble uses all models; no "winner" concept |
| Pydantic models for specs | **Simplified** | `ForecastConfig` dataclass is lighter, faster |
| Full covariate system | **Simplified** | Basic covariate support only; reduces complexity |
| Calibration module | **Removed** | Can be added externally if needed; not core to ensemble |
| Anomaly detection | **Removed** | Post-processing concern; keep core minimal |
| Hierarchical reconciliation | **Removed** | Use hierarchicalforecast directly; out of scope |
| Router bucketing | **Removed** | No need for feature-driven model selection |
| Monitoring/Drift | **Removed** | Infrastructure concern; not forecasting core |

### 5.2 What Was Kept (And Simplified)

| Feature | v1.x | v2.0 |
|---|---|---|
| TSFM adapters | Complex adapter system | Thin wrappers (~120 lines each) |
| Statistical baselines | Many models | 3 essentials (Naive, SeasonalNaive, HistoricAverage) |
| Data validation | Full schema validation | Essential checks only |
| Error codes | 20+ error types | 4 core types with context |
| Pipeline stages | 12+ steps | 6 essential stages |

### 5.3 What Was Added

| Feature | Benefit |
|---|---|
| Ensemble aggregation | Robustness, simplicity, no selection bias |
| `_ensemble_count` column | Transparency—know how many models contributed |
| `require_all_tsfm` option | Strict mode for TSFM-only deployments |
| Lazy adapter loading | Faster imports when TSFM not used |

---

## 6. Constraints & Guardrails

### 6.1 Temporal Integrity (Preserved)
- No random splits
- All features PIT-safe (left-closed windows)
- Covariate leakage detection (if covariates used)

### 6.2 Ensemble Guarantees
- At least `min_models_for_ensemble` must succeed
- If `require_all_tsfm=True`, any TSFM failure aborts
- Statistical models are "failsafe"—they always work

### 6.3 Determinism
- Same config + data → same ensemble behavior
- Model availability affects which models participate (documented)

---

## 7. Verification & Acceptance Criteria

### 7.1 Must-Have (Blocking)

A1. **Ensemble correctness**: Median/mean of predictions matches expected element-wise aggregation
A2. **Model participation**: All successful models contribute to ensemble
A3. **TSFM availability**: `tsfm_mode="required"` raises error if no adapters available
A4. **Graceful degradation**: Works with statistical-only when TSFM unavailable (preferred mode)
A5. **Temporal integrity**: No future leakage in covariates or features
A6. **Minimum models**: Respects `min_models_for_ensemble` threshold

### 7.2 Should-Have (Quality)

B1. **Performance**: Ensemble overhead <50% of single model fit time (parallel fitting)
B2. **Transparency**: `_ensemble_count` correctly reports participating models
B3. **Error clarity**: Failed models reported in `model_errors` with clear messages

---

## 8. Migration from v1.x

### 8.1 API Changes

```python
# v1.x
from tsagentkit import TaskSpec, run_forecast
spec = TaskSpec(h=7, freq="D")
result = run_forecast(data, spec)

# v2.0
from tsagentkit import forecast
result = forecast(data, h=7, freq="D")
```

### 8.2 Behavior Changes

| Aspect | v1.x | v2.0 |
|---|---|---|
| Model selection | Best model per series | Ensemble of all models |
| Backtesting | Competitive, selects winner | Optional validation only |
| Output model name | Winning model (e.g., "SeasonalNaive") | "ensemble_median" |
| TSFM policy | Preferred/required/disabled | Same, but all TSFMs participate |

### 8.3 Recommended Migration Path

1. Replace `TaskSpec` with direct `forecast()` parameters
2. Update expectations: no per-series model winners
3. Use `result.forecast.df["_ensemble_count"]` for transparency
4. Remove competitive backtest logic from surrounding code

---

## 9. Future Extensions (Out of Scope for v2.0)

These features may be added in v2.1+ without breaking core ensemble paradigm:

- **Weighted ensemble**: Weight models by historical performance
- **Calibration**: Post-process ensemble intervals for coverage
- **Hierarchical reconciliation**: Post-ensemble coherence adjustment
- **Feature engineering**: Deterministic feature blocks before ensemble

---

## 10. Appendix: Design Decision Log

| Decision | Alternative | Rationale |
|---|---|---|
| Ensemble over selection | Competitive backtesting | Simpler, more robust, faster |
| Median default | Mean | Median is robust to outlier models |
| Dataclass config | Pydantic models | Faster, simpler, no JSON schema need |
| 3 statistical models | 10+ models | 80% of value with 20% of complexity |
| Remove calibration | Include in core | Calibration can be added externally |
| Remove hierarchy | Include in core | Use hierarchicalforecast directly |
| Lazy adapter loading | Eager loading | Faster startup when TSFM not used |

---

**Summary**: v2.0 trades feature breadth for depth and simplicity. The ensemble approach is a principled simplification that maintains robustness while dramatically reducing code size and cognitive load.
