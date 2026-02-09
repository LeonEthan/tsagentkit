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
  --storage-path ./data \
  --output-path ./results
```

## 5. Analyze Results

```bash
uv run python analyze.py --results-file ./results/all_results.csv
```

## 6. Validate and Package Submission

Strict full-run validation (expects 98 rows):

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
