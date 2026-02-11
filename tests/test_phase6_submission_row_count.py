"""Phase 6 guard: submission expected rows must match dataset matrix."""

from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_assignment_literal(path: Path, name: str) -> object:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment '{name}' not found in {path}")


def test_submission_default_expected_rows_matches_dataset_matrix() -> None:
    root = _repo_root()
    data_loader = root / "benchmarks" / "gift_eval" / "eval" / "data_loader.py"
    submission = root / "benchmarks" / "gift_eval" / "eval" / "submission.py"

    short_datasets = _read_assignment_literal(data_loader, "SHORT_DATASETS")
    med_long_datasets = _read_assignment_literal(data_loader, "MED_LONG_DATASETS")
    expected_rows = _read_assignment_literal(submission, "DEFAULT_EXPECTED_ROWS")

    computed = len(short_datasets) + 2 * len(med_long_datasets)
    assert expected_rows == computed
