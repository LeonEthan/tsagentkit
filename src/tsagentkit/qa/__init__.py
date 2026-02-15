"""QA checks for tsagentkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class QAReport:
    """Quality assurance report."""

    issues: list[dict[str, Any]] = field(default_factory=list)
    valid: bool = True

    def has_critical_issues(self) -> bool:
        return any(issue.get("severity") == "critical" for issue in self.issues)


def run_qa(
    data: pd.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    min_train_size: int = 56,
) -> QAReport:
    """Run basic QA checks on data."""
    issues = []

    # Check series lengths
    series_lengths = data.groupby(id_col).size()
    short_series = series_lengths[series_lengths < min_train_size]
    if len(short_series) > 0:
        issues.append({
            "type": "min_history",
            "count": len(short_series),
            "severity": "warning",
        })

    # Check for duplicates
    dups = data.duplicated(subset=[id_col, time_col], keep=False).sum()
    if dups > 0:
        issues.append({
            "type": "duplicates",
            "count": int(dups),
            "severity": "critical",
        })

    # Check temporal ordering
    for uid, group in data.groupby(id_col):
        if not group[time_col].is_monotonic_increasing:
            issues.append({
                "type": "not_monotonic",
                "series": uid,
                "severity": "critical",
            })

    return QAReport(
        issues=issues,
        valid=not any(i.get("severity") == "critical" for i in issues),
    )


__all__ = ["QAReport", "run_qa"]
