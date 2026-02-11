"""Phase 4 tests for benchmark session/model-pool integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_PREDICTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "gift_eval"
    / "eval"
    / "predictor.py"
)
_SPEC = importlib.util.spec_from_file_location("gift_eval_predictor", _PREDICTOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
predictor_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(predictor_mod)


def _fake_panel_and_meta() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    panel_df = pd.DataFrame(
        {
            "unique_id": ["uid-1", "uid-1", "uid-1"],
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        }
    )
    metadata = {
        "uid-1": {
            "item_id": "item-1",
            "fcst_start": pd.Period("2024-01-04", freq="D"),
        }
    }
    return panel_df, metadata


def _fake_run_result(horizon: int) -> SimpleNamespace:
    forecast_df = pd.DataFrame(
        {
            "unique_id": ["uid-1"] * horizon,
            "ds": pd.date_range("2024-01-04", periods=horizon, freq="D"),
            "yhat": [10.0 + i for i in range(horizon)],
        }
    )
    return SimpleNamespace(forecast=SimpleNamespace(df=forecast_df))


def test_predict_batch_uses_session_and_pool_fit_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_session(self) -> None:
        self._model_pool = object()
        self._session = "session-object"

    def fake_run_forecast(**kwargs):
        captured.update(kwargs)
        return _fake_run_result(horizon=2)

    monkeypatch.setattr(predictor_mod.TSAgentKitPredictor, "_init_session", fake_init_session)
    monkeypatch.setattr(predictor_mod.TSAgentKitPredictor, "_to_df", lambda self, batch: _fake_panel_and_meta())
    monkeypatch.setattr(predictor_mod, "run_forecast", fake_run_forecast)

    predictor = predictor_mod.TSAgentKitPredictor(mode="quick", batch_size=1)
    outputs = predictor._predict_batch(batch=[{}], h=2, freq="D")

    assert len(outputs) == 1
    assert captured["mode"] == "quick"
    assert captured["session"] == "session-object"
    assert callable(captured["fit_func"])


def test_pool_backed_fit_returns_artifact_with_pooled_adapter() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.fit_calls = 0

        def fit(self, dataset, prediction_length: int, quantiles):  # noqa: ANN001
            _ = dataset
            _ = prediction_length
            _ = quantiles
            self.fit_calls += 1

    class FakePool:
        def __init__(self, adapter) -> None:  # noqa: ANN001
            self.adapter = adapter
            self.requested: list[str] = []

        def get(self, adapter_name: str):  # noqa: ANN201
            self.requested.append(adapter_name)
            return self.adapter

    predictor = predictor_mod.TSAgentKitPredictor(
        mode="quick",
        batch_size=1,
    )
    adapter = FakeAdapter()
    pool = FakePool(adapter)
    predictor._model_pool = pool

    dataset = SimpleNamespace(task_spec=SimpleNamespace(horizon=5, season_length=1))
    plan = SimpleNamespace(candidate_models=["tsfm-chronos"], quantiles=[0.1, 0.5])

    artifact = predictor._fit_with_model_pool(dataset=dataset, plan=plan)

    assert pool.requested == ["chronos"]
    assert adapter.fit_calls == 1
    assert artifact.model is adapter
    assert artifact.model_name == "tsfm-chronos"
    assert artifact.metadata["model_pool"] is True
