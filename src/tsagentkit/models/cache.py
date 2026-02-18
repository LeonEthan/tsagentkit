"""Model cache for TSFM singleton lifecycle management.

TSFMs (Time-Series Foundation Models) have large parameters (100MB-2GB).
Loading them on every forecast() call is prohibitively expensive.

The ModelCache keeps loaded models in memory for reuse across calls.
"""

from __future__ import annotations

import gc
import importlib
from typing import Any

from tsagentkit.models.registry import REGISTRY, ModelSpec


class ModelCache:
    """Singleton cache for loaded TSFM models.

    TSFMs are expensive to load but cheap to predict.
    Cache keeps them in memory for reuse across calls.

    Example:
        # Automatic caching (standard pipeline)
        result = forecast(df, h=7)  # Models cached automatically

        # Explicit control (batch processing)
        ModelCache.preload([REGISTRY["chronos"], REGISTRY["timesfm"]])
        for df in batch:
            result = forecast(df, h=7)  # Uses cached models
        ModelCache.unload()  # Free memory
    """

    _cache: dict[str, Any] = {}

    @classmethod
    def get(cls, spec: ModelSpec, device: str | None = None) -> Any:
        """Get cached model or load if not exists.

        Args:
            spec: Model specification
            device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)

        Returns:
            Loaded model instance
        """
        if spec.name not in cls._cache:
            cls._cache[spec.name] = cls._load(spec, device=device)
        return cls._cache[spec.name]

    @classmethod
    def preload(cls, models: list[ModelSpec], device: str | None = None) -> None:
        """Pre-load multiple models (useful for batch processing).

        Args:
            models: List of model specifications to load
            device: Device to load models on ('cuda', 'mps', 'cpu', or None for auto)
        """
        for spec in models:
            if spec.name not in cls._cache:
                cls._cache[spec.name] = cls._load(spec, device=device)

    @classmethod
    def unload(cls, model_name: str | None = None) -> None:
        """Unload model(s) to free memory.

        Args:
            model_name: Specific model to unload, or None to clear all

        Notes:
            This releases all tsagentkit-owned references and runs backend cache
            cleanup. If external code keeps references to model objects, Python
            cannot reclaim that memory until those references are dropped.
        """
        names = [model_name] if model_name is not None else list(cls._cache.keys())

        for name in names:
            model = cls._cache.pop(name, None)
            if model is not None:
                cls._unload_adapter(name, model)

        cls._release_backend_memory()

    @classmethod
    def list_loaded(cls) -> list[str]:
        """List currently cached models.

        Returns:
            List of cached model names
        """
        return list(cls._cache.keys())

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        """Check if a model is currently cached.

        Args:
            model_name: Model name to check

        Returns:
            True if model is in cache
        """
        return model_name in cls._cache

    @classmethod
    def _load(cls, spec: ModelSpec, device: str | None = None) -> Any:
        """Load model from adapter.

        Args:
            spec: Model specification
            device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)

        Returns:
            Loaded model instance
        """
        module = importlib.import_module(spec.adapter_path)
        load_fn = getattr(module, "load")
        config = dict(spec.config_fields)
        if device is not None:
            config["device"] = device
        return load_fn(**config)

    @classmethod
    def _unload_adapter(cls, model_name: str, model: Any) -> None:
        """Call adapter unload hook if available."""
        spec = REGISTRY.get(model_name)
        if spec is None:
            return

        module = importlib.import_module(spec.adapter_path)
        unload_fn = getattr(module, "unload", None)
        if unload_fn is None:
            return

        try:
            unload_fn(model)
        except TypeError:
            unload_fn()

    @classmethod
    def _release_backend_memory(cls) -> None:
        """Best-effort backend cache cleanup for released models."""
        gc.collect()

        try:
            import torch
        except ImportError:
            return

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available() and hasattr(cuda, "empty_cache"):
            cuda.empty_cache()
            if hasattr(cuda, "ipc_collect"):
                cuda.ipc_collect()

        backend_mps = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(backend_mps and hasattr(backend_mps, "is_available") and backend_mps.is_available())
        torch_mps = getattr(torch, "mps", None)
        if mps_available and torch_mps is not None and hasattr(torch_mps, "empty_cache"):
            torch_mps.empty_cache()


__all__ = ["ModelCache"]
