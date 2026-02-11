"""Minimal non-mock TSFM smoke gate for CI.

This test is intentionally tiny: it validates that at least one real TSFM
adapter can load and produce a forecast end-to-end.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry
from tsagentkit.series import TSDataset
from tsagentkit.serving import clear_tsfm_cache, get_tsfm_model

pytestmark = [
    pytest.mark.tsfm,
    pytest.mark.skipif(
        os.getenv("TSFM_RUN_REAL") != "1",
        reason="Set TSFM_RUN_REAL=1 to run real adapter smoke gate.",
    ),
]


def _make_dataset(freq: str = "D", periods: int = 20) -> TSDataset:
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * periods + ["B"] * periods,
            "ds": list(pd.date_range("2023-01-01", periods=periods, freq=freq)) * 2,
            "y": list(range(periods)) + list(range(periods, 0, -1)),
        }
    )
    spec = TaskSpec(h=2, freq=freq, tsfm_policy={"mode": "preferred"})
    return TSDataset.from_dataframe(df, spec)


def _candidate_adapters() -> list[tuple[str, dict[str, object]]]:
    return [
        ("chronos", {"model_size": "small", "device": "cpu", "num_samples": 4}),
        ("timesfm", {"model_size": "base", "device": "cpu"}),
        ("moirai", {"model_size": "small", "device": "cpu", "num_samples": 4}),
    ]


def test_real_adapter_minimal_smoke_gate() -> None:
    dataset = _make_dataset()
    selected = os.getenv("TSFM_REAL_SMOKE_ADAPTER", "").strip()
    candidates = _candidate_adapters()
    if selected:
        candidates = [item for item in candidates if item[0] == selected]
        if not candidates:
            pytest.fail(f"Unknown TSFM_REAL_SMOKE_ADAPTER='{selected}'.")

    failures: list[str] = []
    for adapter_name, kwargs in candidates:
        is_available, reason = AdapterRegistry.check_availability(adapter_name)
        if not is_available:
            failures.append(f"{adapter_name}: unavailable ({reason})")
            continue

        try:
            clear_tsfm_cache()
            _adapter = get_tsfm_model(adapter_name, **kwargs)
            # Build config explicitly to validate creation path without mocks.
            _ = AdapterConfig(model_name=adapter_name, **kwargs)
            if not _adapter.is_loaded:
                _adapter.load_model()
            result = _adapter.predict(dataset, horizon=dataset.task_spec.horizon)

            assert not result.df.empty
            assert len(result.df) == dataset.n_series * dataset.task_spec.horizon
            assert "yhat" in result.df.columns
            return
        except Exception as exc:
            failures.append(f"{adapter_name}: runtime failure ({type(exc).__name__}: {exc})")

    joined = "\n".join(failures)
    pytest.fail(f"No real TSFM adapter passed minimal smoke gate.\n{joined}")
