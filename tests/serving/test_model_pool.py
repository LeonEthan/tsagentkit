"""Tests for serving/model_pool.py."""

from __future__ import annotations

import pytest

from tsagentkit.contracts import (
    EAdapterNotAvailable,
    EModelNotLoaded,
    ForecastResult,
    ModelArtifact,
)
from tsagentkit.models.adapters import AdapterRegistry
from tsagentkit.models.adapters.base import TSFMAdapter
from tsagentkit.serving import ModelPool, ModelPoolConfig


class _PoolFakeAdapter(TSFMAdapter):
    load_counts: dict[str, int] = {}
    unload_counts: dict[str, int] = {}

    def load_model(self) -> None:
        name = self.config.model_name
        self._model = {"name": name}
        self.__class__.load_counts[name] = self.__class__.load_counts.get(name, 0) + 1

    def unload_model(self) -> None:
        name = self.config.model_name
        self.__class__.unload_counts[name] = self.__class__.unload_counts.get(name, 0) + 1
        super().unload_model()

    def fit(self, dataset, prediction_length: int, quantiles=None) -> ModelArtifact:
        return ModelArtifact(
            model=self,
            model_name=self.config.model_name,
            config={"prediction_length": prediction_length, "quantiles": quantiles},
        )

    def predict(self, dataset, horizon: int, quantiles=None) -> ForecastResult:
        raise NotImplementedError

    def get_model_signature(self) -> str:
        return f"{self.config.model_name}-{self._device}"

    @classmethod
    def _check_dependencies(cls) -> None:
        return None


def _register_fake_adapters(names: list[str]) -> None:
    for name in names:
        AdapterRegistry.register(name, _PoolFakeAdapter)


def _unregister_fake_adapters(names: list[str]) -> None:
    for name in names:
        AdapterRegistry.unregister(name)


def test_preload_all_respects_max_three_constraint() -> None:
    names = ["poolfake1", "poolfake2", "poolfake3", "poolfake4"]
    _register_fake_adapters(names)
    _PoolFakeAdapter.load_counts.clear()

    try:
        pool = ModelPool(
            ModelPoolConfig(
                adapters=tuple(names),
                preload=True,
                max_preload_adapters=3,
            )
        )
        stats = pool.stats()

        assert stats["loaded_count"] == 3
        assert stats["loaded_adapters"] == names[:3]
        assert sum(_PoolFakeAdapter.load_counts.values()) == 3
    finally:
        _unregister_fake_adapters(names)


def test_get_reuses_same_adapter_instance() -> None:
    names = ["poolfakereuse"]
    _register_fake_adapters(names)
    _PoolFakeAdapter.load_counts.clear()

    try:
        pool = ModelPool(
            ModelPoolConfig(
                adapters=tuple(names),
                preload=False,
                allow_lazy_load_on_miss=True,
            )
        )

        first = pool.get("poolfakereuse")
        second = pool.get("poolfakereuse")

        assert first is second
        assert _PoolFakeAdapter.load_counts["poolfakereuse"] == 1
    finally:
        _unregister_fake_adapters(names)


@pytest.mark.parametrize(
    ("max_preload", "expected_loaded"),
    [
        (1, ["poolmatrix1"]),
        (2, ["poolmatrix1", "poolmatrix2"]),
        (3, ["poolmatrix1", "poolmatrix2", "poolmatrix3"]),
    ],
)
def test_strict_preload_matrix_and_guardrail(
    max_preload: int,
    expected_loaded: list[str],
) -> None:
    names = ["poolmatrix1", "poolmatrix2", "poolmatrix3"]
    _register_fake_adapters(names)
    _PoolFakeAdapter.load_counts.clear()

    try:
        pool = ModelPool(
            ModelPoolConfig(
                adapters=tuple(names),
                preload=True,
                max_preload_adapters=max_preload,
                allow_lazy_load_on_miss=False,
            )
        )

        assert pool.stats()["loaded_adapters"] == expected_loaded
        assert sum(_PoolFakeAdapter.load_counts.values()) == len(expected_loaded)

        for adapter_name in expected_loaded:
            _ = pool.get(adapter_name)

        if max_preload < 3:
            with pytest.raises(EModelNotLoaded):
                pool.get("poolmatrix3")
        with pytest.raises(EAdapterNotAvailable):
            pool.get("poolmatrix4")
    finally:
        _unregister_fake_adapters(names)


def test_close_unloads_and_clears_pool() -> None:
    names = ["poolfakeclose1", "poolfakeclose2"]
    _register_fake_adapters(names)
    _PoolFakeAdapter.load_counts.clear()
    _PoolFakeAdapter.unload_counts.clear()

    try:
        pool = ModelPool(
            ModelPoolConfig(
                adapters=tuple(names),
                preload=True,
            )
        )
        assert pool.stats()["loaded_count"] == 2

        pool.close()
        stats = pool.stats()

        assert stats["loaded_count"] == 0
        assert stats["loaded_adapters"] == []
        assert _PoolFakeAdapter.unload_counts["poolfakeclose1"] == 1
        assert _PoolFakeAdapter.unload_counts["poolfakeclose2"] == 1
    finally:
        _unregister_fake_adapters(names)
