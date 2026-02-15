# tsagentkit v2.0 Architecture

> Purpose: Describe the minimalist v2.0 technical architecture, module boundaries, and the ensemble-first design paradigm.
>
> Scope: Code structure, data flow, and module responsibilities for the simplified v2.0 codebase.
>
> Last Updated: 2026-02-15

---

## 1. Architecture Goals

- **Ensemble by Default**: All models participate; final prediction is aggregated.
- **Minimalist**: <2,000 lines, focused scope, no feature creep.
- **Deterministic**: Same input → same ensemble behavior.
- **Agent-Friendly**: Single entry point, sensible defaults, clear errors.
- **TSFM-First**: Leverage foundation models with statistical baselines as ensemble participants.

---

## 2. Core Design Decision: Ensemble > Selection

### v1.x Approach (Competitive Selection)

```
Input → Validate → Features → Router Bucketing → Competitive Backtest
                                                    ↓
                                          Select Winner per Series
                                                    ↓
                                          Fit Winner → Predict
```

**Problems**: Complex routing logic, per-series model tracking, winner selection heuristics, large codebase (~10K lines).

### v2.0 Approach (Ensemble Aggregation)

```
Input → Validate → QA → Build Ensemble Plan → Fit All Models (parallel)
                                                  ↓
                                        Predict All → Median/Mean Ensemble
                                                  ↓
                                        Package Result
```

**Benefits**:
1. Simpler: No competitive evaluation, no winner selection
2. More robust: Outlier models dampened by aggregation
3. Faster: Parallel model fitting vs sequential competitive CV
4. Smaller: ~1,800 lines vs ~10,000 lines
5. Predictable: Same models run every time

---

## 3. Module Structure (v2.0)

### 3.1 Module Overview

| Module | Responsibility | Key Types | Lines |
|--------|---------------|-----------|-------|
| `core/` | Config, data structures, errors, results | `ForecastConfig`, `TSDataset`, `RunResult`, `ForecastResult` | ~450 |
| `core/config.py` | Unified configuration | `ForecastConfig` | ~155 |
| `core/data.py` | Dataset containers | `TSDataset`, `CovariateSet` | ~108 |
| `core/errors.py` | Error types with context | `TSAgentKitError`, `EModelFailed`, `ETSFMRequired` | ~120 |
| `core/results.py` | Output containers | `ForecastResult`, `RunResult` | ~65 |
| `models/` | Model adapters and fitting | `fit()`, `predict()`, `fit_tsfm()`, `predict_tsfm()` | ~146 |
| `models/adapters/` | TSFM wrappers | `ChronosAdapter`, `TimesFMAdapter`, `MoiraiAdapter` | ~400 |
| `pipeline/` | Pipeline stages and runner | `PipelineStage`, `ensemble_stage()`, `forecast()` | ~483 |
| `router/` | Ensemble plan building | `Plan`, `ModelCandidate`, `build_plan()` | ~167 |
| `qa/` | Basic QA checks | `run_qa()` | ~66 |

**Total**: ~1,800 lines (target: <2,000)

### 3.2 Dependency Direction

```
qa/ ──────────┐
core/ ────────┼──→ pipeline/ (orchestration)
              │
models/ ←─────┘
    ↑
adapters/ (chronos, timesfm, moirai)
```

**Rules**:
- `core/` has no dependencies on other modules
- `models/` depends on `core/`
- `pipeline/` depends on `core/`, `models/`, `router/`, `qa/`
- Adapters are loaded lazily to avoid heavy imports

---

## 4. Data Flow

### 4.1 Pipeline Stages

```python
# Stage 1: Validation
df = validate_stage(data, config)
# - Check required columns
# - Normalize column names
# - Raise EContractViolation on issues

# Stage 2: QA
df = qa_stage(df, config)
# - Check series lengths
# - Detect duplicates
# - Verify temporal ordering
# - Raise EDataQuality on critical issues

# Stage 3: Dataset Building
dataset = build_dataset_stage(df, config, covariates)
# - Create TSDataset
# - Sort by (unique_id, ds)

# Stage 4: Ensemble Plan
dataset, plan = plan_stage(dataset, config)
# - Inspect TSFM adapters
# - Build Plan with tsfm_models + statistical_models
# - Configure ensemble_method, require_all_tsfm

# Stage 5: Ensemble Execution
forecast_result, model_errors = ensemble_stage(dataset, plan, config)
# - Fit all models in plan
# - Collect successful predictions
# - Compute median/mean ensemble
# - Return ForecastResult with _ensemble_count

# Stage 6: Packaging
result = package_stage(forecast_result, model_errors, config, duration_ms)
# - Create RunResult with metadata
```

### 4.2 Key Data Structures

#### ForecastConfig (Central Configuration)
```python
@dataclass(frozen=True)
class ForecastConfig:
    # Core
    h: int
    freq: str = "D"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Ensemble
    ensemble_method: Literal["median", "mean"] = "median"
    require_all_tsfm: bool = False
    min_models_for_ensemble: int = 1

    # TSFM Policy
    tsfm_mode: Literal["required", "preferred", "disabled"] = "preferred"

    # Execution
    mode: Literal["quick", "standard", "strict"] = "standard"
    n_backtest_windows: int = 0  # Disabled in v2.0
```

#### Plan (Ensemble Configuration)
```python
@dataclass(frozen=True)
class Plan:
    tsfm_models: list[ModelCandidate]          # e.g., [chronos, timesfm, moirai]
    statistical_models: list[ModelCandidate]   # e.g., [SeasonalNaive, HistoricAverage, Naive]
    ensemble_method: str                       # "median" or "mean"
    require_all_tsfm: bool
    min_models_for_ensemble: int

    def all_models(self) -> list[ModelCandidate]:
        return self.tsfm_models + self.statistical_models
```

#### ForecastResult (Output)
```python
@dataclass(frozen=True)
class ForecastResult:
    df: pd.DataFrame  # Columns: unique_id, ds, yhat, _ensemble_count, [quantiles]
    model_name: str   # "ensemble_median" or "ensemble_mean"
    config: Any
```

---

## 5. Ensemble Mechanics

### 5.1 Model Fitting

```python
def ensemble_stage(dataset, plan, config):
    predictions = []
    model_errors = []

    for candidate in plan.all_models():
        try:
            if candidate.is_tsfm:
                artifact = fit_tsfm(dataset, candidate.adapter_name)
                pred = predict_tsfm(dataset, artifact, config.h)
            else:
                artifact = fit(dataset, candidate.name)
                pred = predict(dataset, artifact, config.h)
            predictions.append(pred)
        except Exception as e:
            if candidate.is_tsfm and plan.require_all_tsfm:
                raise EModelFailed(f"Required TSFM {candidate.name} failed")
            model_errors.append({"model": candidate.name, "error": str(e)})

    # Check minimum models
    if len(predictions) < plan.min_models_for_ensemble:
        raise EModelFailed("Insufficient models succeeded")

    # Compute ensemble
    ensemble_df = _compute_ensemble(predictions, plan.ensemble_method)
    return ForecastResult(df=ensemble_df, model_name=f"ensemble_{plan.ensemble_method}")
```

### 5.2 Aggregation

```python
def _compute_ensemble(predictions: list[pd.DataFrame], method: str) -> pd.DataFrame:
    """Element-wise median or mean across all predictions."""
    import numpy as np

    base = predictions[0].copy()
    yhat_stack = np.stack([p["yhat"].values for p in predictions])

    if method == "median":
        base["yhat"] = np.median(yhat_stack, axis=0)
    elif method == "mean":
        base["yhat"] = np.mean(yhat_stack, axis=0)

    base["_ensemble_count"] = len(predictions)
    return base
```

### 5.3 TSFM Adapter Protocol

```python
class TSFMAdapter(Protocol):
    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load pretrained model (zero-shot)."""
        ...

    def predict(
        self,
        dataset: TSDataset,
        artifact: dict[str, Any],
        h: int
    ) -> pd.DataFrame:
        """Generate forecasts."""
        ...
```

Adapters are thin wrappers (~120 lines each):
- `chronos.py`: Amazon Chronos2
- `timesfm.py`: Google TimesFM 2.5
- `moirai.py`: Salesforce Moirai

---

## 6. Error Handling

### 6.1 Error Hierarchy

```
TSAgentKitError (base)
├── EContractViolation    # Input validation failures
├── EDataQuality          # Data quality issues
├── EModelFailed          # All models failed or min not met
└── ETSFMRequired         # TSFM required but unavailable
```

### 6.2 Error Context

All errors include:
- `error_code`: Machine-readable identifier
- `message`: Human-readable description
- `context`: Dict with relevant debugging info
- `fix_hint`: Suggested remediation

Example:
```python
raise EModelFailed(
    "Insufficient models succeeded: 0 < 1",
    context={"errors": [("chronos", "ImportError")], "succeeded": 0},
    fix_hint="Install TSFM packages or set tsfm_mode='disabled'"
)
```

---

## 7. API Surface

### 7.1 Primary Entry Point

```python
def forecast(
    data: pd.DataFrame,
    h: int,
    freq: str = "D",
    **kwargs  # Passed to ForecastConfig
) -> RunResult:
    """Single-function forecasting with ensemble."""
    config = ForecastConfig(h=h, freq=freq, **kwargs)
    return run_pipeline(data, config)
```

### 7.2 Presets

```python
# Quick experimentation
ForecastConfig.quick(h=7)
# - ensemble_method="median"
# - tsfm_mode="preferred"
# - mode="quick"

# Standard production
ForecastConfig.standard(h=14)
# - ensemble_method="median"
# - tsfm_mode="required"

# Strict
ForecastConfig.strict(h=14)
# - tsfm_mode="required"
# - require_all_tsfm=True
```

### 7.3 Inspection

```python
def inspect_tsfm_adapters() -> list[str]:
    """Return available TSFM adapters ['chronos', 'timesfm', 'moirai']."""
```

---

## 8. Testing Strategy

### 8.1 Test Structure

```
tests/
└── test_v2_ensemble.py      # 20 comprehensive tests
    ├── TestEnsembleConfig   # Configuration tests
    ├── TestBuildPlan        # Plan construction
    ├── TestComputeEnsemble  # Aggregation logic
    ├── TestIntegration      # End-to-end
    └── TestForecastConfigPresets
```

### 8.2 Key Test Coverage

- Median/mean aggregation correctness
- Multi-series ensemble
- TSFM mode variations (required/preferred/disabled)
- Minimum models enforcement
- Model error tracking
- Configuration presets

---

## 9. Comparison: v1.x vs v2.0

### 9.1 Code Size

| Component | v1.x | v2.0 |
|-----------|------|------|
| Core modules | ~3,000 lines | ~450 lines |
| Models | ~2,000 lines | ~350 lines |
| Adapters | ~1,500 lines | ~400 lines |
| Pipeline | ~2,000 lines | ~550 lines |
| Router | ~1,500 lines | ~170 lines |
| Backtest | ~1,500 lines | Removed |
| Evaluation | ~1,000 lines | Removed |
| Calibration | ~800 lines | Removed |
| Anomaly | ~600 lines | Removed |
| Monitoring | ~700 lines | Removed |
| Hierarchy | ~1,200 lines | Removed |
| **Total** | **~15,800** | **~1,800** |

### 9.2 Feature Comparison

| Feature | v1.x | v2.0 |
|---------|------|------|
| Competitive backtesting | Yes | No (not needed for ensemble) |
| Per-series model selection | Yes | No (ensemble uses all) |
| Ensemble forecasting | No | Yes (default) |
| TSFM adapters | Yes | Yes (simplified) |
| Statistical baselines | 10+ | 3 (Naive, SeasonalNaive, HistoricAverage) |
| Calibration | Yes | No (external) |
| Anomaly detection | Yes | No (external) |
| Hierarchical reconciliation | Yes | No (use hierarchicalforecast) |
| Covariate support | Full | Basic |
| Monitoring | Full | No (infrastructure concern) |

### 9.3 API Comparison

```python
# v1.x
from tsagentkit import TaskSpec, PlanSpec, run_forecast, rolling_backtest
from tsagentkit.router import RouteDecision

spec = TaskSpec(h=7, freq="D")
plan = make_plan(dataset, spec)
backtest = rolling_backtest(dataset, spec, plan)
winner = select_winner(backtest)
result = run_forecast(data, spec)

# v2.0
from tsagentkit import forecast

result = forecast(data, h=7, freq="D")
# That's it.
```

---

## 10. Extension Points

While v2.0 is minimalist, these extension points are preserved:

### 10.1 Adding TSFM Adapters

```python
# Create models/adapters/newmodel.py
class NewModelAdapter:
    def fit(self, dataset): ...
    def predict(self, dataset, artifact, h): ...

# Register in models/__init__.py
adapter_map["newmodel"] = "tsagentkit.models.adapters.newmodel"
```

### 10.2 Custom Ensemble Methods

Extend `_compute_ensemble()` in `pipeline/stages.py`:

```python
if method == "weighted":
    weights = get_model_weights()  # Custom weighting
    base["yhat"] = np.average(yhat_stack, axis=0, weights=weights)
```

### 10.3 Post-Processing

The `RunResult` provides everything needed for external processing:

```python
result = forecast(data, h=7)
forecast_df = result.forecast.df

# External calibration
calibrated = my_calibrator.calibrate(forecast_df)

# External anomaly detection
anomalies = my_detector.detect(calibrated)
```

---

## 11. Design Principles

1. **Ensemble by Default**: All models participate; no selection complexity.
2. **Fail Fast on Principles**: Clear errors for temporal violations.
3. **Graceful on Practicality**: Individual model failures don't break ensemble.
4. **Thin Adapters**: Delegate to mature libraries; don't reimplement.
5. **Functional > OOP**: Pure functions over class hierarchies.
6. **Explicit > Implicit**: `_ensemble_count` shows what contributed.
7. **Minimal > Complete**: Do less, but do it well.

---

## 12. Migration Notes

### For v1.x Users

1. Replace `TaskSpec` with `forecast()` parameters
2. Remove competitive backtesting expectations
3. Update result parsing: check `_ensemble_count` not winner model
4. Move calibration/anomaly detection outside the library
5. Use `hierarchicalforecast` directly for hierarchy needs

### Breaking Changes

- `run_forecast()` → `forecast()`
- `TaskSpec` → `ForecastConfig`
- `RunArtifact` → `RunResult`
- `backtest_report` → removed (was for competitive selection)
- `model_used` now shows "ensemble_median" not specific model name

---

**Summary**: v2.0 is a deliberate simplification that trades feature breadth for clarity and robustness. The ensemble paradigm eliminates an entire class of complexity (model selection) while maintaining or improving forecast quality through aggregation.
