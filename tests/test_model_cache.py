"""Tests for ModelCache lifecycle behavior."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from tsagentkit.models.cache import ModelCache
from tsagentkit.models.registry import REGISTRY, ModelSpec


def _spec(name: str, adapter_path: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        adapter_path=adapter_path,
        config_fields={},
        requires=[],
        is_tsfm=True,
    )


@pytest.fixture(autouse=True)
def clear_model_cache():
    ModelCache._cache.clear()
    yield
    ModelCache._cache.clear()


def test_get_reuses_cached_model(monkeypatch):
    """ModelCache.get should load once then reuse."""
    spec = _spec("dummy_reuse", "dummy.reuse")
    sentinel = object()
    calls = {"count": 0}

    def fake_load(cls, _spec_obj):
        calls["count"] += 1
        return sentinel

    monkeypatch.setattr(ModelCache, "_load", classmethod(fake_load))

    first = ModelCache.get(spec)
    second = ModelCache.get(spec)

    assert first is sentinel
    assert second is sentinel
    assert calls["count"] == 1


def test_unload_single_calls_adapter_unload(monkeypatch):
    """ModelCache.unload(name) should call adapter unload hook."""
    spec = _spec("dummy_single", "dummy.single")
    monkeypatch.setitem(REGISTRY, spec.name, spec)

    model = object()
    ModelCache._cache[spec.name] = model

    seen = {"artifact": None}
    module = SimpleNamespace(unload=lambda artifact=None: seen.update(artifact=artifact))
    monkeypatch.setattr(importlib, "import_module", lambda _path: module)

    releases = {"count": 0}

    def fake_release(cls):
        releases["count"] += 1

    monkeypatch.setattr(ModelCache, "_release_backend_memory", classmethod(fake_release))

    ModelCache.unload(spec.name)

    assert spec.name not in ModelCache._cache
    assert seen["artifact"] is model
    assert releases["count"] == 1


def test_unload_fallbacks_to_zero_arg_adapter_unload(monkeypatch):
    """If adapter unload accepts no args, ModelCache should fallback cleanly."""
    spec = _spec("dummy_noarg", "dummy.noarg")
    monkeypatch.setitem(REGISTRY, spec.name, spec)

    ModelCache._cache[spec.name] = object()

    seen = {"called": 0}

    def noarg_unload():
        seen["called"] += 1

    module = SimpleNamespace(unload=noarg_unload)
    monkeypatch.setattr(importlib, "import_module", lambda _path: module)

    monkeypatch.setattr(
        ModelCache,
        "_release_backend_memory",
        classmethod(lambda cls: None),
    )

    ModelCache.unload(spec.name)

    assert spec.name not in ModelCache._cache
    assert seen["called"] == 1


def test_unload_all_clears_cache_and_calls_hooks(monkeypatch):
    """ModelCache.unload() should clear all cached models."""
    spec_a = _spec("dummy_a", "dummy.a")
    spec_b = _spec("dummy_b", "dummy.b")
    monkeypatch.setitem(REGISTRY, spec_a.name, spec_a)
    monkeypatch.setitem(REGISTRY, spec_b.name, spec_b)

    model_a = object()
    model_b = object()
    ModelCache._cache[spec_a.name] = model_a
    ModelCache._cache[spec_b.name] = model_b

    seen = {"a": None, "b": None, "release": 0}

    modules = {
        "dummy.a": SimpleNamespace(unload=lambda artifact=None: seen.update(a=artifact)),
        "dummy.b": SimpleNamespace(unload=lambda artifact=None: seen.update(b=artifact)),
    }
    monkeypatch.setattr(importlib, "import_module", lambda path: modules[path])

    def fake_release(cls):
        seen["release"] += 1

    monkeypatch.setattr(ModelCache, "_release_backend_memory", classmethod(fake_release))

    ModelCache.unload()

    assert ModelCache.list_loaded() == []
    assert seen["a"] is model_a
    assert seen["b"] is model_b
    assert seen["release"] == 1
