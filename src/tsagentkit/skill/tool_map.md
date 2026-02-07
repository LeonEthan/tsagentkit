# tsagentkit Tool Map

## What
Task-to-API lookup for coding agents building production forecasting systems.

## When
Use this map to choose the minimal stable API surface for each step.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `task_spec`: `TaskSpec`
- Optional: custom model functions, covariates, monitoring config

## Workflow
1. Validate and QA
2. Build dataset and route plan
3. Fit, predict, backtest
4. Package and persist lifecycle artifacts

---

## Stable Assembly APIs
- `validate_contract`
- `run_qa`
- `align_covariates`
- `build_dataset` / `TSDataset.from_dataframe`
- `make_plan`
- `rolling_backtest`
- `fit`
- `predict`
- `package_run`
- `save_run_artifact`
- `load_run_artifact`
- `validate_run_artifact_for_serving`
- `replay_forecast_from_artifact`

## Routing + TSFM APIs
- `make_plan`
- `inspect_tsfm_adapters`
- `build_plan_graph`
- `attach_plan_graph`
- `get_adapter_capability`
- `list_adapter_capabilities`

## Task -> API Mapping
| Task | Primary API | Notes |
|---|---|---|
| Schema validation | `validate_contract` | Fails on missing/invalid columns |
| Quality checks | `run_qa` | `quick/standard/strict` |
| Covariate alignment | `align_covariates` | Leakage-safe policies |
| Dataset construction | `build_dataset` | Immutable `TSDataset` |
| Plan generation | `make_plan` | Deterministic candidates + reasons |
| Plan graph for custom orchestration | `attach_plan_graph` | Explicit DAG-like nodes |
| Adapter capability gating | `list_adapter_capabilities` | Runtime availability included |
| Backtesting | `rolling_backtest` | Temporal-only windows |
| Model fit/predict | `fit`, `predict` | Uses router plan fallback semantics |
| End artifact | `package_run` | Includes provenance and metadata |
| Persist artifact | `save_run_artifact` | JSON payload with schema metadata |
| Load artifact | `load_run_artifact` | Safe reconstruction with checks |
| Serving gate | `validate_run_artifact_for_serving` | Signature/schema compatibility |
| Replay forecast | `replay_forecast_from_artifact` | Deterministic replay view |
| Wrapper pipeline | `run_forecast` | Convenience only, not primary |

## Error Handling Priorities
- Contract failures: `E_CONTRACT_*`
- Temporal guardrails: `E_SPLIT_RANDOM_FORBIDDEN`, `E_DS_NOT_MONOTONIC`
- Leakage guardrails: `E_COVARIATE_LEAKAGE`
- Model execution: `E_MODEL_FIT_FAIL`, `E_MODEL_PREDICT_FAIL`, `E_FALLBACK_EXHAUSTED`
- Lifecycle boundaries: `E_ARTIFACT_SCHEMA_INCOMPATIBLE`, `E_ARTIFACT_LOAD_FAILED`

## Related Docs
- `skill/README.md`
- `skill/recipes.md`
- `docs/API_STABILITY.md`
- `docs/ARCHITECTURE.md`
