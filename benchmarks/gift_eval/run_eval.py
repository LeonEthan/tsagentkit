#!/usr/bin/env python3
"""Run tsagentkit evaluation on GIFT-Eval benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .eval import MED_LONG_DATASETS, SHORT_DATASETS, GIFTEvalRunner, TSAgentKitPredictor
except ImportError:
    from eval import MED_LONG_DATASETS, SHORT_DATASETS, GIFTEvalRunner, TSAgentKitPredictor

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)


def configure_logging(log_file: Path) -> None:
    """Configure file and stream logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def get_all_configs() -> list[tuple[str, str]]:
    """Return all benchmark (dataset, term) configurations."""
    configs: list[tuple[str, str]] = [(ds, "short") for ds in SHORT_DATASETS]
    for ds in MED_LONG_DATASETS:
        configs.extend([(ds, "medium"), (ds, "long")])
    return configs


def load_progress(progress_file: Path) -> dict[str, list]:
    """Load resume progress file if present."""
    if progress_file.exists():
        with progress_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress: dict[str, list], progress_file: Path) -> None:
    """Persist progress state for resume mode."""
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def reset_tsfm_runtime_stats() -> None:
    """Reset runtime TSFM load stats when telemetry is available."""
    try:
        from tsagentkit.models.telemetry import reset_tsfm_runtime_stats as _reset

        _reset()
    except Exception:  # noqa: BLE001
        logger.debug("TSFM telemetry reset unavailable", exc_info=True)


def get_tsfm_runtime_stats() -> dict[str, object]:
    """Read runtime TSFM load stats when telemetry is available."""
    try:
        from tsagentkit.models.telemetry import get_tsfm_runtime_stats as _get

        return _get()
    except Exception:  # noqa: BLE001
        logger.debug("TSFM telemetry snapshot unavailable", exc_info=True)
        return {
            "load_count": 0,
            "load_time_ms_total": 0.0,
            "per_adapter": {},
        }


def append_runtime_stats(runtime_file: Path, payload: dict[str, object]) -> None:
    """Append one runtime-stats record in JSONL format."""
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    with runtime_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tsagentkit on GIFT-Eval")
    parser.add_argument("--dataset", type=str, help="Single dataset to evaluate")
    parser.add_argument("--datasets", type=str, nargs="+", help="Multiple datasets to evaluate")
    parser.add_argument(
        "--term",
        type=str,
        default="short",
        choices=["short", "medium", "long"],
        help="Forecasting term for --dataset or --datasets",
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmark configurations")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["quick", "standard", "strict"],
        help="tsagentkit execution mode",
    )
    parser.add_argument("--storage-path", type=str, default="./data", help="Path to GIFT-Eval data")
    parser.add_argument("--output-path", type=str, default="./results", help="Path to output results")
    parser.add_argument("--batch-size", type=int, default=512, help="Predictor batch size")
    parser.add_argument(
        "--preload-adapters",
        type=str,
        nargs="+",
        default=["chronos"],
        help="Ordered adapters to preload (max 3), e.g. chronos moirai",
    )
    parser.add_argument("--download", action="store_true", help="Download benchmark data only")
    parser.add_argument("--resume", action="store_true", help="Resume from progress.json")
    parser.add_argument("--max-datasets", type=int, help="Limit number of configs (for smoke testing)")
    return parser.parse_args()


def resolve_configs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Resolve dataset/term configurations from CLI args."""
    if args.all:
        configs = get_all_configs()
    elif args.datasets:
        configs = [(name, args.term) for name in args.datasets]
    elif args.dataset:
        configs = [(args.dataset, args.term)]
    else:
        raise ValueError("Specify one of --dataset, --datasets, or --all")

    if args.max_datasets:
        configs = configs[: args.max_datasets]
    return configs


def main() -> int:
    args = parse_args()
    if len(args.preload_adapters) > 3:
        logger.error("--preload-adapters supports up to 3 adapters, got %d", len(args.preload_adapters))
        return 1

    storage_path = Path(args.storage_path).resolve()
    output_path = Path(args.output_path).resolve()
    logs_path = Path("./logs").resolve()
    configure_logging(logs_path / "eval.log")

    output_path.mkdir(parents=True, exist_ok=True)
    os.environ["GIFT_EVAL"] = str(storage_path)

    if args.download:
        GIFTEvalRunner.download_data(storage_path=storage_path)
        logger.info("Download complete")
        return 0

    if not storage_path.exists():
        logger.error("Data not found at %s. Run with --download first.", storage_path)
        return 1

    try:
        datasets_to_run = resolve_configs(args)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    progress_file = output_path / "progress.json"
    if args.resume:
        progress = load_progress(progress_file)
        completed = {tuple(item) for item in progress["completed"]}
        failed = {tuple(item[:2]) for item in progress["failed"]}
        datasets_to_run = [cfg for cfg in datasets_to_run if cfg not in completed]
        logger.info(
            "Resume mode: %d remaining, %d completed",
            len(datasets_to_run),
            len(completed),
        )
    else:
        progress = {"completed": [], "failed": []}
        completed, failed = set(), set()

    logger.info(
        "Running TSAgentKit (%s mode) on %d config(s)",
        args.mode,
        len(datasets_to_run),
    )

    start_time = datetime.now()
    total = len(datasets_to_run)

    shared_predictor: TSAgentKitPredictor | None = None
    try:
        for idx, (dataset_name, term) in enumerate(datasets_to_run, start=1):
            logger.info("%s", "=" * 64)
            logger.info("[%d/%d] %s/%s", idx, total, dataset_name, term)
            logger.info("%s", "=" * 64)

            try:
                reset_tsfm_runtime_stats()
                if shared_predictor is None:
                    shared_predictor = TSAgentKitPredictor(
                        mode=args.mode,
                        batch_size=args.batch_size,
                        preload_adapters=args.preload_adapters,
                    )
                forecast_start = time.perf_counter()
                runner = GIFTEvalRunner(
                    dataset_name=dataset_name,
                    term=term,
                    output_path=output_path,
                    storage_path=storage_path,
                    mode=args.mode,
                    preload_adapters=args.preload_adapters,
                )
                results = runner.evaluate(predictor=shared_predictor, batch_size=args.batch_size)
                forecast_ms = (time.perf_counter() - forecast_start) * 1000.0
                runtime_stats = get_tsfm_runtime_stats()
                load_count = int(runtime_stats.get("load_count", 0))
                load_time_ms_total = float(runtime_stats.get("load_time_ms_total", 0.0))
                per_adapter = runtime_stats.get("per_adapter", {})

                mase = float(results["eval_metrics/MASE[0.5]"].iloc[-1])
                smape = float(results["eval_metrics/sMAPE[0.5]"].iloc[-1])
                crps = float(results["eval_metrics/mean_weighted_sum_quantile_loss"].iloc[-1])
                logger.info("MASE: %.4f | sMAPE: %.4f | CRPS: %.4f", mase, smape, crps)
                logger.info(
                    "Runtime stats | load_count=%d | load_time_ms_total=%.2f | forecast_time_ms_total=%.2f",
                    load_count,
                    load_time_ms_total,
                    forecast_ms,
                )
                if per_adapter:
                    logger.info("Runtime stats (per_adapter): %s", per_adapter)

                append_runtime_stats(
                    output_path / "runtime_stats.jsonl",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dataset_name,
                        "term": term,
                        "mode": args.mode,
                        "batch_size": args.batch_size,
                        "load_count": load_count,
                        "load_time_ms_total": load_time_ms_total,
                        "forecast_time_ms_total": forecast_ms,
                        "per_adapter": per_adapter,
                    },
                )

                completed.add((dataset_name, term))
                progress["completed"].append([dataset_name, term])
                failed.discard((dataset_name, term))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed %s/%s: %s", dataset_name, term, exc)
                failed.add((dataset_name, term))
                progress["failed"].append([dataset_name, term, str(exc)])

            save_progress(progress, progress_file)

            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / idx
            remaining = total - idx
            eta = datetime.now() + pd.Timedelta(seconds=avg_time * remaining)
            logger.info("Progress %d/%d | ETA: %s", idx, total, eta.strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        if shared_predictor is not None:
            shared_predictor.close()

    logger.info("%s", "=" * 64)
    logger.info("EVALUATION COMPLETE")
    logger.info("%s", "=" * 64)
    logger.info("Completed: %d/%d", len(completed), total)
    if failed:
        logger.warning("Failed: %d", len(failed))
        for dataset_name, term in sorted(failed):
            logger.warning("  - %s/%s", dataset_name, term)

    return 0


if __name__ == "__main__":
    sys.exit(main())
