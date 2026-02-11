"""Session-oriented TSFM model pool.

Phase 1 scope:
- immutable pool config
- bounded preload (max 3 adapters)
- deterministic preload order
- process-local adapter reuse via get()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from tsagentkit.contracts import EAdapterNotAvailable, EModelNotLoaded
from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tsagentkit.models.adapters import TSFMAdapter


@dataclass(frozen=True)
class ModelPoolConfig:
    """Immutable configuration for ModelPool."""

    adapters: tuple[str, ...] = ("chronos", "moirai", "timesfm")
    model_size_by_adapter: Mapping[str, str] = field(default_factory=dict)
    adapter_kwargs_by_adapter: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    device: str | None = None
    preload: bool = False
    max_preload_adapters: int = 3
    allow_lazy_load_on_miss: bool = False

    def __post_init__(self) -> None:
        if not self.adapters:
            raise ValueError("ModelPoolConfig.adapters must contain at least one adapter name.")
        if self.max_preload_adapters < 1:
            raise ValueError("ModelPoolConfig.max_preload_adapters must be >= 1.")
        if self.max_preload_adapters > 3:
            raise ValueError("ModelPoolConfig.max_preload_adapters cannot exceed 3.")

        object.__setattr__(self, "adapters", tuple(self.adapters))
        object.__setattr__(
            self,
            "model_size_by_adapter",
            MappingProxyType(dict(self.model_size_by_adapter)),
        )
        normalized_kwargs: dict[str, Mapping[str, Any]] = {}
        for name, kwargs in self.adapter_kwargs_by_adapter.items():
            normalized_kwargs[name] = MappingProxyType(dict(kwargs))
        object.__setattr__(
            self,
            "adapter_kwargs_by_adapter",
            MappingProxyType(normalized_kwargs),
        )


class ModelPool:
    """Reusable TSFM adapter pool with optional eager preload."""

    def __init__(self, config: ModelPoolConfig | None = None) -> None:
        self.config = config or ModelPoolConfig()
        self._lock = threading.RLock()
        self._adapters: dict[str, TSFMAdapter] = {}
        self._stats: dict[str, dict[str, Any]] = {}

        if self.config.preload:
            self.preload_all()

    def preload_all(self) -> list[str]:
        """Preload configured adapters up to max_preload_adapters."""
        loaded: list[str] = []
        to_preload = self.config.adapters[: self.config.max_preload_adapters]
        for adapter_name in to_preload:
            self.preload(adapter_name)
            loaded.append(adapter_name)
        return loaded

    def preload(self, adapter_name: str) -> TSFMAdapter:
        """Preload one configured adapter and return it."""
        with self._lock:
            if adapter_name in self._adapters:
                return self._adapters[adapter_name]
            self._ensure_configured(adapter_name)
            adapter = self._create_loaded_adapter(adapter_name)
            self._adapters[adapter_name] = adapter
            return adapter

    def get(self, adapter_name: str) -> TSFMAdapter:
        """Get adapter from pool.

        In strict mode (default), miss on a non-preloaded adapter raises
        an explicit error instead of lazy loading.
        """
        with self._lock:
            if adapter_name in self._adapters:
                return self._adapters[adapter_name]
            self._ensure_configured(adapter_name)
            if not self.config.allow_lazy_load_on_miss:
                raise EModelNotLoaded(
                    f"Adapter '{adapter_name}' is configured but not preloaded in this session.",
                    context={
                        "adapter_name": adapter_name,
                        "configured_adapters": list(self.config.adapters),
                        "loaded_adapters": list(self._adapters.keys()),
                        "max_preload_adapters": self.config.max_preload_adapters,
                    },
                    fix_hint=(
                        "Include adapter in preload list and init with preload=True, "
                        "or explicitly set allow_lazy_load_on_miss=True."
                    ),
                )

            adapter = self._create_loaded_adapter(adapter_name)
            self._adapters[adapter_name] = adapter
            return adapter

    def close(self) -> None:
        """Unload and remove all pooled adapters."""
        with self._lock:
            for adapter in self._adapters.values():
                try:
                    adapter.unload_model()
                except Exception:  # noqa: BLE001
                    # Best-effort cleanup.
                    continue
            self._adapters.clear()
            self._stats.clear()

    def stats(self) -> dict[str, Any]:
        """Return pool statistics snapshot."""
        with self._lock:
            return {
                "configured_adapters": list(self.config.adapters),
                "max_preload_adapters": self.config.max_preload_adapters,
                "allow_lazy_load_on_miss": self.config.allow_lazy_load_on_miss,
                "loaded_adapters": list(self._adapters.keys()),
                "loaded_count": len(self._adapters),
                "details": {
                    name: {
                        "model_size": detail["model_size"],
                        "device": detail["device"],
                        "load_time_ms": detail["load_time_ms"],
                    }
                    for name, detail in self._stats.items()
                },
            }

    def _ensure_configured(self, adapter_name: str) -> None:
        if adapter_name not in self.config.adapters:
            raise EAdapterNotAvailable(
                f"Adapter '{adapter_name}' is not configured in this ModelPool session.",
                context={
                    "adapter_name": adapter_name,
                    "configured_adapters": list(self.config.adapters),
                },
                fix_hint="Add adapter to ModelPoolConfig.adapters before session initialization.",
            )

    def _build_adapter_config(self, adapter_name: str) -> AdapterConfig:
        extra_kwargs = dict(self.config.adapter_kwargs_by_adapter.get(adapter_name, {}))
        model_size = self.config.model_size_by_adapter.get(adapter_name)
        if model_size is None:
            model_size = str(extra_kwargs.pop("model_size", extra_kwargs.pop("pipeline", "base")))
        device = self.config.device if self.config.device is not None else extra_kwargs.pop("device", None)
        extra_kwargs.pop("model_name", None)
        return AdapterConfig(
            model_name=adapter_name,
            model_size=model_size,
            device=device,
            **extra_kwargs,
        )

    def _create_loaded_adapter(self, adapter_name: str) -> TSFMAdapter:
        adapter_config = self._build_adapter_config(adapter_name)
        adapter_class = AdapterRegistry.get(adapter_name)

        adapter = adapter_class(adapter_config)
        start = time.perf_counter()
        adapter.load_model()
        load_time_ms = (time.perf_counter() - start) * 1000.0

        self._stats[adapter_name] = {
            "model_size": adapter_config.model_size,
            "device": adapter_config.device,
            "load_time_ms": load_time_ms,
        }
        return adapter


__all__ = ["ModelPool", "ModelPoolConfig"]
