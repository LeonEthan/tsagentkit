# tsagentkit Tool Map

## What
Quick mapping of common tasks to the recommended tsagentkit entry points.

## When
Use this to choose the correct module or function when building workflows.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `task_spec`: `TaskSpec`
- Optional: `fit_func` (fit(dataset, plan)), `predict_func` (predict(dataset, artifact, spec)), `monitoring_config`

## Workflow
- Validate schema and ordering: `validate_contract`
- Run QA checks/repairs: `run_qa`
- Build dataset: `TSDataset.from_dataframe` or `build_dataset`
- Plan model routing: `make_plan`
- Run backtest: `rolling_backtest`
- Fit and predict: `fit` + `predict`
- Full pipeline: `run_forecast`
