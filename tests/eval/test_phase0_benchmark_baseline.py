"""Phase 0 baseline tests for benchmark-oriented TSFM loading behavior."""

from __future__ import annotations

import pandas as pd

from tsagentkit import TaskSpec
from tsagentkit.contracts import ForecastResult, ModelArtifact
from tsagentkit.models.adapters import AdapterRegistry
from tsagentkit.models.adapters.base import TSFMAdapter, _timed_model_load
from tsagentkit.models.telemetry import (
    get_tsfm_runtime_stats,
    reset_tsfm_runtime_stats,
)
from tsagentkit.serving import run_forecast


class _Phase0FakeAdapter(TSFMAdapter):
    load_calls = 0

    @_timed_model_load
    def load_model(self) -> None:
        self._model = {"loaded": True}
        _Phase0FakeAdapter.load_calls += 1

    def _prepare_model(
        self,
        dataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> dict[str, object]:
        return {}

    def _get_model_name(self) -> str:
        return f"{self.config.model_name}-model"

    def predict(
        self,
        dataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        if not self.is_loaded:
            self.load_model()

        rows: list[dict[str, object]] = []
        offset = pd.tseries.frequencies.to_offset(dataset.freq)
        for uid in dataset.series_ids:
            series = dataset.get_series(uid)
            last_date = series["ds"].max()
            for step in range(1, horizon + 1):
                rows.append(
                    {
                        "unique_id": uid,
                        "ds": last_date + step * offset,
                        "yhat": float(step),
                        "model": self.config.model_name,
                    }
                )

        return ForecastResult(
            df=pd.DataFrame(rows),
            provenance=self._create_provenance(dataset, horizon, quantiles),
            model_name=self.config.model_name,
            horizon=horizon,
        )

    def get_model_signature(self) -> str:
        return f"{self.config.model_name}-{self._device}"

    @classmethod
    def _check_dependencies_impl(cls) -> None:
        pass

    @classmethod
    def _get_capability_spec(cls, adapter_name: str) -> dict[str, object]:
        return {
            "adapter_name": adapter_name,
            "provider": "test",
            "is_zero_shot": True,
            "supports_quantiles": True,
            "supports_past_covariates": False,
            "supports_future_covariates": False,
            "supports_static_covariates": False,
            "max_context_length": None,
            "max_horizon": None,
            "dependencies": [],
            "notes": "Fake adapter for testing.",
        }


def test_single_dataset_smoke_fixture_is_deterministic(
    deterministic_single_dataset_smoke_config,
    deterministic_single_dataset_smoke_panel: pd.DataFrame,
) -> None:
    assert deterministic_single_dataset_smoke_config == {
        "dataset": "m4_hourly",
        "term": "short",
        "mode": "quick",
        "batch_size": 32,
    }
    assert list(deterministic_single_dataset_smoke_panel.columns) == ["unique_id", "ds", "y"]
    assert len(deterministic_single_dataset_smoke_panel) == 24
    assert deterministic_single_dataset_smoke_panel["unique_id"].nunique() == 2


def test_repeated_run_forecast_triggers_repeated_tsfm_loads(
    deterministic_single_dataset_smoke_panel: pd.DataFrame,
) -> None:
    adapter_name = "phase0fake"
    AdapterRegistry.register(adapter_name, _Phase0FakeAdapter)

    reset_tsfm_runtime_stats()
    _Phase0FakeAdapter.load_calls = 0

    try:
        spec = TaskSpec(
            h=3,
            freq="D",
            tsfm_policy={"mode": "required", "adapters": [adapter_name]},
        )

        first = run_forecast(
            data=deterministic_single_dataset_smoke_panel,
            task_spec=spec,
            mode="quick",
        )
        second = run_forecast(
            data=deterministic_single_dataset_smoke_panel,
            task_spec=spec,
            mode="quick",
        )

        assert not first.forecast.df.empty
        assert not second.forecast.df.empty

        stats = get_tsfm_runtime_stats()
        assert _Phase0FakeAdapter.load_calls == 2
        assert stats["load_count"] == 2
        assert stats["per_adapter"][adapter_name]["load_count"] == 2
        # Timing is measured by decorator, just verify it's positive
        assert stats["per_adapter"][adapter_name]["load_time_ms_total"] > 0
    finally:
        AdapterRegistry.unregister(adapter_name)

