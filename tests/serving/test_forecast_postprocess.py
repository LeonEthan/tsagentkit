from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from tsagentkit.serving.forecast_postprocess import (
    add_model_column,
    maybe_reconcile_forecast,
    normalize_and_sort_forecast,
    postprocess_forecast,
    resolve_model_name,
)


def test_resolve_model_name_prefers_explicit_field() -> None:
    artifact = SimpleNamespace(model_name="Naive", metadata={"model_name": "Other"})
    assert resolve_model_name(artifact) == "Naive"


def test_resolve_model_name_falls_back_to_metadata() -> None:
    artifact = SimpleNamespace(metadata={"model_name": "SeasonalNaive"})
    assert resolve_model_name(artifact) == "SeasonalNaive"


def test_add_model_column_injects_when_missing() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "yhat": [1.0],
        }
    )
    updated = add_model_column(forecast, "Naive")
    assert "model" in updated.columns
    assert updated["model"].tolist() == ["Naive"]


def test_add_model_column_keeps_existing_values() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "yhat": [1.0],
            "model": ["existing"],
        }
    )
    updated = add_model_column(forecast, "Naive")
    assert updated["model"].tolist() == ["existing"]


def test_normalize_and_sort_forecast_characterization() -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["B", "A", "A"],
            "ds": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-01"]),
            "yhat": [2.0, 3.0, 1.0],
            "q10": [2.2, 3.2, None],
            "q_0.1": [None, None, 1.2],
        }
    )

    normalized = normalize_and_sort_forecast(forecast)
    assert list(normalized["unique_id"]) == ["A", "A", "B"]
    assert "q0.1" in normalized.columns
    assert "q10" not in normalized.columns
    assert "q_0.1" not in normalized.columns
    assert normalized["q0.1"].tolist() == [1.2, 3.2, 2.2]


def test_maybe_reconcile_forecast_applies_only_when_plan_present(monkeypatch) -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "yhat": [1.0],
        }
    )

    class _HierDataset:
        hierarchy = object()

        @staticmethod
        def is_hierarchical() -> bool:
            return True

    calls: list[str] = []

    def _fake_apply_reconciliation_if_needed(
        forecast: pd.DataFrame,
        hierarchy: object,
        method: str,
    ) -> pd.DataFrame:
        _ = hierarchy
        calls.append(method)
        result = forecast.copy()
        result["reconciled"] = True
        return result

    monkeypatch.setattr(
        "tsagentkit.hierarchy.apply_reconciliation_if_needed",
        _fake_apply_reconciliation_if_needed,
    )

    untouched = maybe_reconcile_forecast(
        forecast,
        dataset=_HierDataset(),
        plan=None,
        reconciliation_method="bottom_up",
    )
    assert "reconciled" not in untouched.columns

    reconciled = maybe_reconcile_forecast(
        forecast,
        dataset=_HierDataset(),
        plan=object(),
        reconciliation_method="min_trace",
    )
    assert reconciled["reconciled"].tolist() == [True]
    assert calls == ["min_trace"]


def test_postprocess_forecast_combines_steps(monkeypatch) -> None:
    forecast = pd.DataFrame(
        {
            "unique_id": ["B", "A"],
            "ds": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "yhat": [2.0, 1.0],
        }
    )

    class _HierDataset:
        hierarchy = object()

        @staticmethod
        def is_hierarchical() -> bool:
            return True

    def _fake_apply_reconciliation_if_needed(
        forecast: pd.DataFrame,
        hierarchy: object,
        method: str,
    ) -> pd.DataFrame:
        _ = hierarchy
        _ = method
        result = forecast.copy()
        result["yhat"] = result["yhat"] + 1.0
        return result

    monkeypatch.setattr(
        "tsagentkit.hierarchy.apply_reconciliation_if_needed",
        _fake_apply_reconciliation_if_needed,
    )

    result = postprocess_forecast(
        forecast,
        model_name="Naive",
        dataset=_HierDataset(),
        plan=object(),
        reconciliation_method="bottom_up",
    )
    assert result["model"].tolist() == ["Naive", "Naive"]
    assert list(result["unique_id"]) == ["A", "B"]
    assert result["yhat"].tolist() == [2.0, 3.0]
