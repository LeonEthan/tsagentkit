from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def deterministic_single_dataset_smoke_config() -> dict[str, object]:
    """Deterministic single-dataset benchmark smoke config."""
    return {
        "dataset": "m4_hourly",
        "term": "short",
        "mode": "quick",
        "batch_size": 32,
    }


@pytest.fixture
def deterministic_single_dataset_smoke_panel() -> pd.DataFrame:
    """Deterministic panel fixture for single-dataset smoke-style tests."""
    periods = 12
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {
            "unique_id": ["A"] * periods + ["B"] * periods,
            "ds": list(dates) * 2,
            "y": list(range(1, periods + 1)) + list(range(periods, 0, -1)),
        }
    )

