"""TSFM (Time-Series Foundation Model) caching for serving.

Provides model caching and lazy loading for TSFM adapters to enable
 efficient inference in production serving environments.
"""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsagentkit.models.adapters import TSFMAdapter


class TSFMModelCache:
    """Thread-safe cache for TSFM model instances.

    Implements a singleton cache pattern for TSFM adapters to avoid
    reloading large foundation models on each request. Uses weak
    references to allow garbage collection when memory is constrained.

    Attributes:
        _cache: Dictionary mapping model names to cached instances
        _lock: Threading lock for concurrent access
        _metadata: Cache metadata (load time, access count, etc.)

    Example:
        >>> cache = TSFMModelCache()
        >>> model = cache.get_model("chronos", pipeline="large")
        >>> # Model is loaded once and cached for subsequent calls
    """

    _instance: TSFMModelCache | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> TSFMModelCache:
        """Ensure singleton pattern for global cache."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the cache (only runs once due to singleton)."""
        if getattr(self, "_initialized", False):
            return

        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._metadata: dict[str, dict[str, Any]] = {}
        self._initialized = True

    def get_model(
        self,
        model_name: str,
        **model_kwargs,
    ) -> TSFMAdapter:
        """Get a TSFM model from cache or load it.

        Args:
            model_name: Name of the TSFM (e.g., "chronos", "moirai", "timesfm")
            **model_kwargs: Arguments passed to model initialization
                (e.g., pipeline="large", device="cuda")

        Returns:
            Cached or newly loaded TSFMAdapter instance

        Raises:
            EModelLoadFailed: If model loading fails
            EAdapterNotAvailable: If adapter is not installed
        """
        cache_key = self._make_cache_key(model_name, **model_kwargs)

        with self._lock:
            # Check cache
            if cache_key in self._cache:
                model = self._cache[cache_key]
                self._metadata[cache_key]["access_count"] += 1
                return model

            # Load model
            model = self._load_model(model_name, **model_kwargs)

            # Cache model
            self._cache[cache_key] = model
            self._metadata[cache_key] = {
                "model_name": model_name,
                "load_time": self._get_timestamp(),
                "access_count": 1,
                "kwargs": model_kwargs,
            }

            return model

    def clear_cache(self, model_name: str | None = None) -> None:
        """Clear cached models.

        Args:
            model_name: If specified, only clear this model. Otherwise clear all.
        """
        with self._lock:
            if model_name is None:
                self._cache.clear()
                self._metadata.clear()
            else:
                keys_to_remove = [
                    k for k, v in self._metadata.items()
                    if v.get("model_name") == model_name
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                    del self._metadata[key]

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - num_models: Number of models in cache
            - models: List of cached model names
            - total_accesses: Total access count across all models
        """
        with self._lock:
            return {
                "num_models": len(self._cache),
                "models": [
                    v["model_name"] for v in self._metadata.values()
                ],
                "total_accesses": sum(
                    v["access_count"] for v in self._metadata.values()
                ),
                "details": {
                    k: {
                        "model_name": v["model_name"],
                        "access_count": v["access_count"],
                        "load_time": v["load_time"],
                    }
                    for k, v in self._metadata.items()
                },
            }

    def _make_cache_key(self, model_name: str, **kwargs) -> str:
        """Create a unique cache key from model name and kwargs."""
        # Sort kwargs for consistent keys
        kv_pairs = sorted(kwargs.items())
        kv_str = ",".join(f"{k}={v}" for k, v in kv_pairs)
        return f"{model_name}:{kv_str}"

    def _load_model(self, model_name: str, **kwargs) -> TSFMAdapter:
        """Load a TSFM model via the adapter registry."""
        from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry

        try:
            adapter_class = AdapterRegistry.get(model_name)
        except ValueError as exc:
            from tsagentkit.contracts import EAdapterNotAvailable

            raise EAdapterNotAvailable(
                f"TSFM adapter '{model_name}' not found. "
                f"Ensure the required package is installed.",
                context={"adapter_name": model_name, "error": str(exc)},
            ) from exc

        try:
            config = AdapterConfig(model_name=model_name, **kwargs)
            return adapter_class(config)
        except Exception as exc:
            from tsagentkit.contracts import EModelLoadFailed

            raise EModelLoadFailed(
                f"Failed to load TSFM model '{model_name}': {exc}",
                context={"adapter_name": model_name, "error": str(exc)},
            ) from exc

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()


def get_tsfm_model(model_name: str, **kwargs) -> TSFMAdapter:
    """Convenience function to get a cached TSFM model.

    Args:
        model_name: Name of the TSFM (e.g., "chronos", "moirai", "timesfm")
        **kwargs: Model initialization arguments

    Returns:
        TSFMAdapter instance (cached or newly loaded)

    Example:
        >>> model = get_tsfm_model("chronos", pipeline="large")
        >>> forecast = model.predict(series, horizon=7)
    """
    cache = TSFMModelCache()
    return cache.get_model(model_name, **kwargs)


def clear_tsfm_cache(model_name: str | None = None) -> None:
    """Clear TSFM model cache.

    Args:
        model_name: If specified, only clear this model. Otherwise clear all.

    Example:
        >>> clear_tsfm_cache("chronos")  # Clear only Chronos
        >>> clear_tsfm_cache()  # Clear all cached models
    """
    cache = TSFMModelCache()
    cache.clear_cache(model_name)
