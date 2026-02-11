"""Phase 6 guard: full matrix row count must match dataset lists."""

from __future__ import annotations

from tsagentkit.gift_eval.data import FULL_MATRIX_SIZE, MED_LONG_DATASETS, SHORT_DATASETS


def test_full_matrix_size_matches_dataset_matrix() -> None:
    computed = len(SHORT_DATASETS) + 2 * len(MED_LONG_DATASETS)
    assert computed == FULL_MATRIX_SIZE
    assert computed == 97
