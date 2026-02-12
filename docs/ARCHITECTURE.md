# tsagentkit Architecture

> Purpose: describe the **implemented technical architecture**, module boundaries and dependency directions, extension points, and stable APIs as an implementation-oriented supplement to the PRD.
> Scope: code structure, data flow, and module responsibilities. Model selection and business policy details are out of scope.

---

## 1. Architecture Goals

- Preserve **determinism** and auditability in time-series forecasting workflows.
- Keep **clear module boundaries and dependency direction** to reduce coupling.
- Provide **explicit extension points** (models, features, calibration, anomaly detection).
- Support **coding-agent orchestration** with stable APIs and strict contracts.

## 2. Layering and Dependency Direction

Recommended one-way dependency stack (bottom to top):

1. `contracts/`, `errors/`
2. `series/`, `time/`, `covariates/`, `features/`
3. `router/`, `backtest/`, `eval/`, `calibration/`, `anomaly/`
4. `models/`
5. `serving/`

Dependency rules:
- Upper layers may depend on lower layers; lower layers must not depend on upper layers.
- `contracts/` must not depend on business modules (Pydantic and stdlib only).
- `serving/` is the only orchestration layer.

Hard constraints:
1. `contracts/`, `series/`, `time/`, `covariates/`, and `features/` must not depend on `serving/` or `models/`.
2. `eval/`, `backtest/`, `router/`, `calibration/`, and `anomaly/` must not depend on `serving/`.
3. Do not reimplement these in-house in this library: statistical/baseline model algorithms, hierarchy reconciliation algorithms, generic metric function implementations.
4. `hierarchy/` accepts only `S_df`/`tags` as canonical inputs and should only perform contract validation and data conversion.
5. `eval/` is a wrapper around `utilsforecast.evaluate`; metric functions must not be duplicated in this repo.
6. `features/` defaults to `tsfeatures`; `tsfresh` is only in `features/extra`.
7. `models/` should only adapt and map parameters; it should not copy `statsforecast`/`sktime` algorithm implementations.
8. `serving/` is a pipeline packaging layer. Public guidance remains assembly-first step composition; `run_forecast()` is a compatibility wrapper.

## 3. Pipeline Data Flow (Implementation View)

Recommended integration is step-level composition. `run_forecast()` wraps the same stable step APIs:

1. `validate_contract()` -> structure and type validation
2. `run_qa()` -> quality checks and optional repair
3. `align_covariates()` -> covariate alignment and leakage checks
4. `TSDataset.from_dataframe()` -> unified dataset object
5. `make_plan()` -> produce `PlanSpec` with competitively assembled candidate pool (TSFM mandatory + statistical models via feature analysis)
6. `build_plan_graph()` / `attach_plan_graph()` -> expose orchestratable DAG nodes (optional)
7. `rolling_backtest()` -> competitive CV evaluation; all candidates evaluated per unique_id for model selection
8. `models.fit()` -> model adaptation and training
9. `models.predict()` -> forecast output
10. `fit_calibrator()` / `apply_calibrator()` -> uncertainty calibration (optional)
11. `detect_anomalies()` -> anomaly detection (optional)
12. `package_run()` -> package `RunArtifact`

Backtesting note:
Backtesting serves as the competitive arena for per-series model selection. All candidates from `PlanSpec` (TSFM mandatory + feature-selected statistical models) are evaluated across rolling windows. The best-performing model per `unique_id` is selected for final prediction. Metrics and summaries should preferentially reuse `utilsforecast.evaluate`.

### Flow Diagram

```mermaid
flowchart TD
    A[Input Panel Data] --> B[Validate Contract]
    B --> C[QA Checks and Repairs]
    C --> D[Align Covariates]
    D --> E[Build TSDataset]
    E --> F[Make Plan]
    F --> G[Competitive Backtest<br/>(all candidates per unique_id)]
    G --> H[Fit Winner<br/>(selected per unique_id)]
    H --> I[Predict]
    I --> J[Calibrate (optional)]
    J --> K[Anomaly (optional)]
    K --> L[Package RunArtifact]
```

## 3.1 Competitive Backtesting Architecture

The backtesting layer operates as a **competitive model selection arena** rather than a simple validation mechanism:

### Candidate Pool Assembly (`router/`)
- **TSFM Mandatory**: At least one TSFM adapter must be available; raises `E_TSFM_REQUIRED_UNAVAILABLE` otherwise
- **Statistical Models via Feature Analysis**: Based on `RouteDecision.stats` (seasonality, intermittency, trend, sparsity), the router assembles appropriate statistical models:
  - Short history → Naive family
  - Intermittent demand → Croston, IMAPA
  - Strong seasonality → SeasonalNaive
  - High frequency → Robust baselines
  - Sparse data → Simple aggregations

### Competitive Evaluation (`backtest/`)
- All candidates evaluated across all rolling CV folds
- Per-series metric aggregation (default: `MASE`)
- Winner selection: `unique_id` → best model mapping

### Final Fit & Predict (`models/`, `serving/`)
- Winner model fitted on full history per `unique_id`
- Prediction uses the fitted winner model
- `RunArtifact` records the selection decision and competitive rankings

## 4. Core Module Responsibilities

- `contracts/`: input/output contracts and config models such as `TaskSpec`/`PlanSpec`
- `series/`: unified `TSDataset`, sparsity analysis, time alignment
- `time/`: frequency inference and future index generation
- `covariates/`: covariate typing, coverage checks, leakage checks
- `features/`: repeatable feature engineering and signatures
- `router/`: deterministic routing, feature-driven candidate pool assembly (TSFM mandatory + statistical models), and per-series model selection policy
- `models/`: unified adapters for TSFM/StatsForecast/Sktime
- `backtest/`: competitive rolling CV evaluation; all candidates evaluated per unique_id; winner selection based on aggregated metrics
- `eval/`: metric computation and aggregation
- `calibration/`: interval/quantile calibration
- `anomaly/`: anomaly scoring and detection
- `serving/`: orchestration, artifact packaging, structured logging, and session runtime (`TSAgentSession` + `ModelPool`)

Serving runtime note:
- `ModelPool` is the canonical TSFM lifecycle manager for session-oriented runs.
- `TSFMModelCache` remains a compatibility utility and delegates model loading through the same adapter-loading path.

## 5. Maintainability and Reuse Strategy (Ecosystem Alignment)

Core principle: **reuse mature libraries first and keep thin adapters** to avoid reimplementing algorithms.

Recommended defaults:
- `statsforecast`: default statistical/baseline model runtime and time-series CV support.
- `utilsforecast`: default `evaluate` metrics path plus preprocessing helpers.
- `hierarchicalforecast`: default hierarchy forecasting and reconciliation runtime using `S_df` + `tags`.
- `tsfeatures`: default statistical feature extraction from `unique_id`/`ds`/`y` panel data.
- `sktime`: optional extension path for classical model ecosystem.
- `tsfresh`: optional heavy feature extraction path (non-default).

Contract-driven data alignment:
- `utilsforecast.evaluate` and preprocessing APIs align with `unique_id`/`ds`/`y` and `PanelContract`.
- `tsfeatures` uses `unique_id`/`ds`/`y` panel format and frequency-aware feature extraction.
- `hierarchicalforecast` uses `S_df` + `tags` as hierarchy constraints and aligns with `ForecastResult` integration.

Default library mapping:

| Module | Default Runtime | Notes |
| --- | --- | --- |
| `models/` | `statsforecast` | Statistical/baseline models and CV |
| `eval/` | `utilsforecast` | Metric functions and evaluation flow |
| `hierarchy/` | `hierarchicalforecast` | Hierarchy forecasting and reconciliation |
| `features/` | `tsfeatures` | Statistical features (default) |
| `features/extra` | `tsfresh` (optional) | High-dimensional feature extraction |
| `models/sktime` | `sktime` (optional) | Classical model extension |

Replacement strategy:
- Move statistical/baseline model logic to `statsforecast`; keep only thin adapters.
- Delegate hierarchy reconciliation to `hierarchicalforecast`; keep lightweight wrappers for conversion and validation.
- Route evaluation metrics through `utilsforecast.evaluate`; avoid custom metric maintenance.
- Prefer `tsfeatures` for default feature extraction; use `tsfresh` only when needed.
- Use `sktime` for classical model special cases.

Integration constraints:
- `models/` and `features/` are **thin adapter layers**, not algorithm hosts.
- Non-core capabilities should be **optional extras**.
- Keep `ForecastContract` and `RunArtifact` stable even if implementation backends change.
- Reuse priority: mature implementation > lightweight wrapper > custom implementation (only when no suitable library exists or strict PIT needs require it).

## 6. Extension Points

### Model Extension
- Register TSFM adapters via `models/adapters`.
- Integrate classical models via `models/sktime` or baseline modules.
- `get_adapter_capability()` / `list_adapter_capabilities()` expose capability + runtime availability for agent policy branching.

### Feature Extension
- `features/` exposes `FeatureConfig` and `FeatureFactory`, defaulting to `tsfeatures`.
- Feature hashes support auditability and reproducibility.

### Monitoring and Hierarchy
- `monitoring/`: drift and stability detection.
- `hierarchy/` and `reconciliation/`: hierarchy forecasting and coherence handling (default runtime is `hierarchicalforecast` using `S_df`/`tags`).

## 7. Data Contracts and Artifacts

### Contracts
- `PanelContract`: `unique_id`, `ds`, `y`
- `ForecastContract`: `model`, `yhat`, quantiles/intervals

### Hierarchy Constraint Inputs
`hierarchicalforecast` represents hierarchy constraints with `S_df` and `tags`; `tsagentkit` treats this as the canonical hierarchy input contract.

### Hierarchy Input Contract Details (`S_df` / `tags` / `Y_hat_df`)
- `S_df`: hierarchy summation matrix DataFrame; row index is all series `unique_id`, columns are bottom-level `unique_id`, and values are aggregation weights (typically 0/1).
- `tags`: dictionary keyed by hierarchy level, values are `unique_id` lists for that level.
- `Y_hat_df`: base forecast DataFrame including `unique_id`, `ds`, and model columns.
- `Y_df`: optional historical DataFrame including `unique_id`, `ds`, `y` for methods that require calibration.

### Backtest Output Standard (`CVFrame` / `BacktestReport`)
- `CVFrame` (long format): `unique_id`, `ds`, `cutoff`, `model`, `y`, `yhat`, optional `q_*`
- `BacktestReport`: at least `cv_frame`, `metrics`, `summary`, `errors`, `n_windows`, `horizon`, plus:
  - `model_rankings`: per-series competitive ranking matrix (model × metric)
  - `selection_decision`: `unique_id` → winning model mapping with rationale
  - `feature_analysis`: statistics used to assemble candidate pool
- Backtest metrics should use `utilsforecast.evaluate`; internal pivot-to-wide transformation is acceptable for computation.
- Competitive evaluation: all `PlanSpec.candidate_models` evaluated; winner selected per `unique_id`

### Evaluation Output Standard (`MetricFrame` / `ScoreSummary`)
- `MetricFrame` (long format): optional `unique_id`, optional `cutoff`, `model`, `metric`, `value`
- `ScoreSummary`: aggregated `model`, `metric`, `value`
- `utilsforecast.evaluate` returns a wide metric table; `tsagentkit` normalizes it into long format for persistence.

### Feature Matrix Contract (`FeatureMatrix`)
- `FeatureMatrix.data` must include `unique_id`, `ds`, `y`, and feature columns.
- `feature_cols` are feature column names, typically from `tsfeatures` outputs.
- To avoid naming collisions with covariates/target, feature columns may use a `tsf_` prefix.
- Default adapter implementation is recommended in `features/tsfeatures_adapter.py`, with `FeatureFactory` as the entrypoint.

### Primary Artifacts
- `ForecastResult`: forecast output + provenance
- `RunArtifact`: complete run package (reports + metadata)

## 8. Stable vs Internal APIs

The following should be treated as **stable APIs** (assembly-first primary path):

- `validate_contract()`
- `run_qa()`
- `align_covariates()`
- `TSDataset.from_dataframe()` / `build_dataset()`
- `make_plan()`
- `build_plan_graph()` / `attach_plan_graph()`
- `rolling_backtest()`
- `models.fit()` / `models.predict()`
- `get_adapter_capability()` / `list_adapter_capabilities()`
- `package_run()`
- `save_run_artifact()` / `load_run_artifact()`
- `validate_run_artifact_for_serving()` / `replay_forecast_from_artifact()`
- `TaskSpec`
- `fit_calibrator()` / `apply_calibrator()`
- `detect_anomalies()`

Compatibility API:
- `run_forecast()` (convenience wrapper, stable semantics)
- `TSFMModelCache` / `get_tsfm_model()` / `clear_tsfm_cache()` (compatibility helpers)

Compatibility boundary (must remain stable):
- Step-level composition semantics and `package_run()` artifact field semantics
- `run_forecast()` wrapper semantics and returned `RunArtifact` field names
- Required columns and names in `PanelContract` / `ForecastContract`
- Minimum required columns in `ForecastResult.df`: `unique_id`, `ds`, `model`, `yhat`

Allowed changes (while preserving the above contracts):
- Internal `BacktestReport` field structures and aggregation details
- Row ordering and additive fields in `MetricFrame` / `ScoreSummary`
- Internal candidate model list and routing rules in `PlanSpec` (without changing external contracts)

## 9. Directory Layout (Implementation)

- `src/tsagentkit/`: library source code
- `docs/PRD.md`: product requirements and constraints
- `docs/ARCHITECTURE.md`: this architecture document

## 10. Versioning and Compatibility Strategy (Recommended)

- Introduce breaking API changes only in minor/major version updates.
- `RunArtifact` and `TaskSpec` have highest compatibility priority.

## 11. Follow-Up Recommendations

- Persist `RouteDecision` as part of the packaged artifact.
- Expand step-level examples and tests to prevent drift back to wrapper-first docs.
- Further decouple `contracts/` and `results/`.

## 12. Migration and Deprecation List (Maintainability)

- Gradually deprecate in-house reconciliation/evaluation logic in `hierarchy/` and replace with `hierarchicalforecast` adapters.
- Gradually deprecate in-house metrics in `backtest/metrics.py` and `eval`, and migrate to `utilsforecast.evaluate`.
- Gradually converge in-house feature logic under `features/` toward `tsfeatures` defaults.
- Keep `HierarchyStructure` as a compatibility layer, but treat `S_df`/`tags` as the canonical internal authority.
