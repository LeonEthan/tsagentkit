# GIFT-Eval Runbook

This runbook defines the operational flow for isolated benchmark runs in `benchmarks/gift_eval`.

## 1. Environment

```bash
cd benchmarks/gift_eval
uv sync --python 3.11
```

## 2. Data Setup

```bash
uv run python run_eval.py --download --storage-path ./data
```

## 3. Smoke Benchmark

```bash
uv run python run_eval.py \
  --dataset m4_hourly \
  --term short \
  --mode quick \
  --max-datasets 1 \
  --preload-adapters chronos \
  --storage-path ./data \
  --output-path ./results
```

## 4. Full Benchmark

```bash
uv run python run_eval.py \
  --all \
  --mode standard \
  --resume \
  --batch-size 512 \
  --preload-adapters chronos moirai \
  --storage-path ./data \
  --output-path ./results
```

## 5. Analyze Results

```bash
uv run python analyze.py --results-file ./results/all_results.csv
```

## 6. Validate and Package Submission

Strict full-run validation (expects 97 rows with the current dataset matrix):

```bash
uv run python prepare_submission.py \
  --validate-only \
  --results-file ./results/all_results.csv
```

Create submission package:

```bash
uv run python prepare_submission.py \
  --results-file ./results/all_results.csv \
  --output-path ./submissions \
  --model-name TSAgentKit-standard \
  --model-type agentic \
  --model-dtype float32 \
  --model-link https://example.com/model \
  --code-link https://example.com/code \
  --org YourOrg \
  --testdata-leakage No \
  --replication-code-available Yes
```

For smoke-only artifacts, add `--allow-partial`.

## 7. Runtime Policy Notes

- Benchmark runtime is process-scoped and reuses one `TSAgentKitPredictor` across datasets.
- TSFM models are preloaded once via `ModelPool` (max 3 adapters).
- If GPU memory is constrained:
  - Start with `--preload-adapters chronos`
  - Add `moirai` only when needed for fallback coverage
  - Add `timesfm` only if budget allows
- `runtime_stats.jsonl` is the source of truth for load count/time reductions.
