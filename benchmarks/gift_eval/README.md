# GIFT-Eval Isolated Workspace

This folder is an isolated benchmark workspace for running GIFT-Eval against the local `tsagentkit` source tree.

## Scope

- Keep benchmark code, dependencies, and runtime artifacts under `benchmarks/gift_eval/`.
- Do not add benchmark dependencies to the repository root `pyproject.toml`.
- Keep downloaded datasets and benchmark outputs out of git.

## Layout

```text
benchmarks/gift_eval/
  pyproject.toml
  .gitignore
  README.md
  RUNBOOK.md
  run_eval.py
  analyze.py
  prepare_submission.py
  eval/
    __init__.py
    data_loader.py
    predictor.py
    runner.py
    dataset_properties.json
  data/
  results/
  logs/
  submissions/
```

## Setup

```bash
cd benchmarks/gift_eval
uv sync
```

## Commands

Download benchmark data:

```bash
uv run python run_eval.py --download --storage-path ./data
```

Run one dataset:

```bash
uv run python run_eval.py --dataset m4_hourly --term short --mode standard
```

Run all configurations with resume:

```bash
uv run python run_eval.py --all --mode standard --resume --batch-size 512
```

Analyze results:

```bash
uv run python analyze.py --results-file ./results/all_results.csv
```

Validate leaderboard schema (requires full run by default):

```bash
uv run python prepare_submission.py --validate-only --results-file ./results/all_results.csv
```

Create submission package (`submissions/<model>/all_results.csv` + `config.json`):

```bash
uv run python prepare_submission.py \
  --results-file ./results/all_results.csv \
  --model-name TSAgentKit-standard \
  --model-type agentic \
  --model-dtype float32 \
  --model-link https://example.com/model \
  --code-link https://example.com/code \
  --org YourOrg \
  --testdata-leakage No \
  --replication-code-available Yes
```

For smoke runs (non-98-row results), add `--allow-partial`.

Runbook:

```bash
cat RUNBOOK.md
```

## Note

- No project-level CI/testing is configured for GIFT-Eval in this repository.
- `benchmarks/gift_eval` remains code-only tooling so you can run benchmark tasks manually when needed.

## Status

- Phase 1: implemented
- Phase 2: implemented
- Phase 4: implemented
- Phase 5: implemented
