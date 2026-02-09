#!/usr/bin/env python3
"""Analyze GIFT-Eval run outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def calculate_overall_score(df: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate benchmark metrics."""
    mase = df["eval_metrics/MASE[0.5]"]
    smape = df["eval_metrics/sMAPE[0.5]"]
    crps = df["eval_metrics/mean_weighted_sum_quantile_loss"]
    return {
        "MASE_mean": float(mase.mean()),
        "MASE_median": float(mase.median()),
        "sMAPE_mean": float(smape.mean()),
        "sMAPE_median": float(smape.median()),
        "CRPS_mean": float(crps.mean()),
        "CRPS_median": float(crps.median()),
    }


def generate_report(df: pd.DataFrame) -> str:
    """Generate a human-readable summary report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("TSAgentKit GIFT-Eval Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 70)

    overall = calculate_overall_score(df)
    for metric, value in overall.items():
        lines.append(f"  {metric}: {value:.4f}")
    lines.append(f"\nTotal datasets: {len(df)}")
    lines.append("")

    lines.append("TOP 5 (lowest CRPS)")
    lines.append("-" * 70)
    best = df.nsmallest(5, "eval_metrics/mean_weighted_sum_quantile_loss")
    for _, row in best.iterrows():
        lines.append(
            f"  {row['dataset']}: CRPS={row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f}, "
            f"MASE={row['eval_metrics/MASE[0.5]']:.4f}"
        )
    lines.append("")

    lines.append("BOTTOM 5 (highest CRPS)")
    lines.append("-" * 70)
    worst = df.nlargest(5, "eval_metrics/mean_weighted_sum_quantile_loss")
    for _, row in worst.iterrows():
        lines.append(
            f"  {row['dataset']}: CRPS={row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f}, "
            f"MASE={row['eval_metrics/MASE[0.5]']:.4f}"
        )
    lines.append("")

    lines.append("PERFORMANCE BY DOMAIN")
    lines.append("-" * 70)
    domain_stats = (
        df.groupby("domain")
        .agg(
            {
                "eval_metrics/MASE[0.5]": ["mean", "count"],
                "eval_metrics/mean_weighted_sum_quantile_loss": "mean",
            }
        )
        .round(4)
    )
    lines.append(domain_stats.to_string())
    lines.append("")

    lines.append("ALL RESULTS")
    lines.append("-" * 70)
    for _, row in df.iterrows():
        lines.append(
            f"{row['dataset']}: "
            f"MASE={row['eval_metrics/MASE[0.5]']:.4f}, "
            f"CRPS={row['eval_metrics/mean_weighted_sum_quantile_loss']:.4f}"
        )
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Analyze GIFT-Eval benchmark results")
    parser.add_argument("--results-file", type=str, default="./results/all_results.csv")
    parser.add_argument("--output-path", type=str, default="./results")
    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        logger.error("Results file not found: %s", results_file)
        return 1

    df = pd.read_csv(results_file)
    logger.info("Loaded %d rows from %s", len(df), results_file)

    report_text = generate_report(df)
    report_file = output_path / "analysis_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Report saved to %s", report_file)

    analysis_payload = {
        "overall": calculate_overall_score(df),
        "results": df.to_dict(orient="records"),
    }
    analysis_file = output_path / "analysis.json"
    with analysis_file.open("w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)
    logger.info("JSON saved to %s", analysis_file)

    print(f"\n{report_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
