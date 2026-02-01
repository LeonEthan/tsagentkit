from __future__ import annotations

import pandas as pd

from tsagentkit.utils import (
    extract_quantiles,
    normalize_quantile_columns,
    parse_quantile_column,
    quantile_col_name,
)


def test_parse_quantile_column() -> None:
    assert parse_quantile_column("q0.1") == 0.1
    assert parse_quantile_column("q_0.1") == 0.1
    assert parse_quantile_column("q10") == 0.1
    assert parse_quantile_column("q_10") == 0.1
    assert parse_quantile_column("q0.05") == 0.05
    assert parse_quantile_column("q0") is None
    assert parse_quantile_column("q1") is None
    assert parse_quantile_column("q100") is None
    assert parse_quantile_column("not_q") is None


def test_extract_quantiles() -> None:
    columns = ["q10", "q_0.1", "q0.9", "yhat", "foo"]
    assert extract_quantiles(columns) == [0.1, 0.9]


def test_quantile_col_name() -> None:
    assert quantile_col_name(0.1) == "q0.1"
    assert quantile_col_name(0.05) == "q0.05"


def test_normalize_quantile_columns() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "yhat": [1.0, 2.0],
            "q10": [1.1, None],
            "q_0.1": [2.2, 2.3],
        }
    )

    normalized = normalize_quantile_columns(df)

    assert "q0.1" in normalized.columns
    assert "q10" not in normalized.columns
    assert "q_0.1" not in normalized.columns
    assert normalized["q0.1"].tolist() == [1.1, 2.3]
