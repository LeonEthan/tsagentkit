"""Model cache for TSFM singleton lifecycle management.

TSFMs (Time-Series Foundation Models) have large parameters (100MB-2GB).
Loading them on every forecast() call is prohibitively expensive.

The ModelCache keeps loaded models in memory for reuse across calls.
"""

from __future__ import annotations

from typing import Any

from tsagentkit.models.registry import ModelSpec


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
    def get(cls, spec: ModelSpec) -> Any:
        """Get cached model or load if not exists.

        Args:
            spec: Model specification

        Returns:
            Loaded model instance
        """
        if spec.name not in cls._cache:
            cls._cache[spec.name] = cls._load(spec)
        return cls._cache[spec.name]

    @classmethod
    def preload(cls, models: list[ModelSpec]) -> None:
        """Pre-load multiple models (useful for batch processing).

        Args:
            models: List of model specifications to load
        """
        for spec in models:
            if spec.name not in cls._cache:
                cls._cache[spec.name] = cls._load(spec)

    @classmethod
    def unload(cls, model_name: str | None = None) -> None:
        """Unload model(s) to free memory.

        Args:
            model_name: Specific model to unload, or None to clear all
        """
        if model_name is None:
            cls._cache.clear()
        elif model_name in cls._cache:
            del cls._cache[model_name]

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
    def _load(cls, spec: ModelSpec) -> Any:
        """Load model from adapter.

        Args:
            spec: Model specification

        Returns:
            Loaded model instance
        """
        # Import adapter module dynamically
        import importlib

        module = importlib.import_module(spec.adapter_path)
        load_fn = getattr(module, "load")
        return load_fn(**spec.config_fields)


__all__ = ["ModelCache"]
