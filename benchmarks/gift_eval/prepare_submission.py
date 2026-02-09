#!/usr/bin/env python3
"""Validate results and package a GIFT-Eval submission artifact."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval import (
    ALLOWED_MODEL_TYPES,
    DEFAULT_EXPECTED_ROWS,
    SubmissionValidationError,
    build_config_payload,
    prepare_submission,
    validate_results_csv,
)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GIFT-Eval submission package")
    parser.add_argument("--results-file", type=str, default="./results/all_results.csv")
    parser.add_argument("--output-path", type=str, default="./submissions")

    parser.add_argument("--model-name", type=str, help="Model identifier for config.json")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=sorted(ALLOWED_MODEL_TYPES),
        help="Model category expected by leaderboard",
    )
    parser.add_argument("--model-dtype", type=str, help="Model dtype, e.g. float32 or bfloat16")
    parser.add_argument("--model-link", type=str, help="Public model URL")
    parser.add_argument("--code-link", type=str, help="Public replication code URL")
    parser.add_argument("--org", type=str, help="Organization/team name")
    parser.add_argument(
        "--testdata-leakage",
        type=str,
        choices=["Yes", "No"],
        help="Whether test data leakage exists",
    )
    parser.add_argument(
        "--replication-code-available",
        type=str,
        choices=["Yes", "No"],
        help="Whether replication code is publicly available",
    )

    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow non-full runs (row count different from expected full benchmark).",
    )
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate results CSV and exit.",
    )
    return parser.parse_args()


def _require_submission_fields(args: argparse.Namespace) -> None:
    required = [
        "model_name",
        "model_type",
        "model_dtype",
        "model_link",
        "code_link",
        "org",
        "testdata_leakage",
        "replication_code_available",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SubmissionValidationError(
            "Missing required submission flags: " + ", ".join(f"--{x.replace('_', '-')}" for x in missing)
        )


def main() -> int:
    configure_logging()
    args = parse_args()

    require_full_rows = not args.allow_partial
    results_path = Path(args.results_file)
    output_path = Path(args.output_path)

    try:
        df = validate_results_csv(
            results_path,
            require_full_rows=require_full_rows,
            expected_rows=args.expected_rows,
        )
        logger.info("Results validation passed: %d rows in %s", len(df), results_path)

        if args.validate_only:
            return 0

        _require_submission_fields(args)
        config_payload = build_config_payload(
            model_name=args.model_name,
            model_type=args.model_type,
            model_dtype=args.model_dtype,
            model_link=args.model_link,
            code_link=args.code_link,
            org=args.org,
            testdata_leakage=args.testdata_leakage,
            replication_code_available=args.replication_code_available,
        )
        csv_target, cfg_target = prepare_submission(
            results_file=results_path,
            output_path=output_path,
            config_payload=config_payload,
            require_full_rows=require_full_rows,
            expected_rows=args.expected_rows,
            overwrite=args.overwrite,
        )
        logger.info("Submission CSV: %s", csv_target)
        logger.info("Submission config: %s", cfg_target)
        return 0
    except SubmissionValidationError as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
