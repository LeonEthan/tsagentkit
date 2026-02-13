"""Base adapter class for Time-Series Foundation Models.

Provides a unified interface for integrating external TSFMs like Chronos,
Moirai, and TimesFM with the tsagentkit pipeline.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:

    from tsagentkit.contracts import (
        AdapterCapabilitySpec,
        ForecastResult,
        ModelArtifact,
        Provenance,
    )
    from tsagentkit.series import TSDataset

F = TypeVar("F", bound=Callable[..., Any])


def _timed_model_load(load_fn: F) -> F:
    """Decorator to time model loading and record telemetry.

    Wraps load_model() implementations to automatically track load time
    and report to the telemetry system.
    """
    @wraps(load_fn)
    def wrapper(self: TSFMAdapter, *args: Any, **kwargs: Any) -> Any:
        from tsagentkit.models.telemetry import record_tsfm_model_load

        start = time.perf_counter()
        try:
            return load_fn(self, *args, **kwargs)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            record_tsfm_model_load(self.config.model_name, duration_ms)
    return wrapper  # type: ignore[return-value]


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for TSFM adapter.

    Attributes:
        model_name: Name of the model/adapter
        model_size: Model size variant (small, base, large)
        device: Compute device (cuda, mps, cpu, or None for auto)
        cache_dir: Directory for model caching
        batch_size: Batch size for training/fitting
        prediction_batch_size: Batch size for prediction
        quantile_method: Method for quantile prediction (sample, direct)
        num_samples: Number of samples for probabilistic forecasting
        max_context_length: Maximum context length the model accepts
    """

    model_name: str
    model_size: Literal["tiny", "small", "base", "large"] = "base"
    device: str | None = None  # Auto-detect if None
    cache_dir: str | None = None
    batch_size: int = 32
    prediction_batch_size: int = 100
    quantile_method: Literal["sample", "direct"] = "sample"
    num_samples: int = 100
    max_context_length: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_sizes = {"tiny", "small", "base", "large"}
        if self.model_size not in valid_sizes:
            raise ValueError(
                f"Invalid model_size '{self.model_size}'. "
                f"Must be one of: {valid_sizes}"
            )

        valid_methods = {"sample", "direct"}
        if self.quantile_method not in valid_methods:
            raise ValueError(
                f"Invalid quantile_method '{self.quantile_method}'. "
                f"Must be one of: {valid_methods}"
            )


class TSFMAdapter(ABC):
    """Abstract base class for Time-Series Foundation Model adapters.

    Provides unified interface for different TSFMs while handling
    model-specific quirks internally. Subclasses implement specific
    adapters for models like Chronos, Moirai, and TimesFM.

    Example:
        >>> config = AdapterConfig(model_name="chronos", model_size="base")
        >>> adapter = ChronosAdapter(config)
        >>> adapter.load_model()
        >>> result = adapter.predict(dataset, horizon=30)

    Attributes:
        config: Adapter configuration
        _model: The underlying model instance (lazy loaded)
        _device: Resolved compute device
    """

    def __init__(self, config: AdapterConfig):
        """Initialize adapter with configuration.

        Args:
            config: Adapter configuration
        """
        self.config = config
        self._model: Any | None = None
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        """Resolve compute device (cuda/mps/cpu).

        Returns the best available device based on hardware support.
        Priority: cuda > mps > cpu

        Returns:
            Device string for PyTorch
        """
        if self.config.device:
            return self.config.device

        # Try to import torch for device detection
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    @abstractmethod
    def load_model(self) -> None:
        """Load the foundation model with caching.

        Downloads and caches the model if not already present.
        Should be called before fit() or predict().

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def _prepare_model(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """Prepare model-specific configuration for prediction.

        Subclasses implement this to perform any model-specific preparation
        (e.g., compiling the model, setting up contexts).

        Args:
            dataset: Training dataset
            prediction_length: Forecast horizon
            quantiles: Optional quantile levels

        Returns:
            Dictionary with model-specific configuration to include in ModelArtifact
        """
        pass

    def fit(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ModelArtifact:
        """Prepare model for prediction on the dataset.

        Note: Most TSFMs are zero-shot and don't require traditional fitting.
        This method validates compatibility and may perform preprocessing.

        Args:
            dataset: Training dataset
            prediction_length: Forecast horizon
            quantiles: Optional quantile levels for probabilistic forecasts

        Returns:
            ModelArtifact with model reference and configuration

        Raises:
            ValueError: If dataset is incompatible with model
            RuntimeError: If preparation fails
        """
        from tsagentkit.contracts import ModelArtifact

        self._require_loaded("fit")
        self._validate_dataset(dataset)

        extra_config = self._prepare_model(dataset, prediction_length, quantiles)

        return ModelArtifact(
            model=self._model,
            model_name=self._get_model_name(),
            config={
                "model_size": self.config.model_size,
                "device": self._device,
                "prediction_length": prediction_length,
                "quantiles": quantiles,
                **extra_config,
            },
        )

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name for ModelArtifact and provenance.

        Returns:
            Model name string (e.g., "chronos-base", "moirai-2.0")
        """
        pass

    @abstractmethod
    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using the TSFM.

        Args:
            dataset: Historical data for context
            horizon: Number of steps to forecast
            quantiles: Optional quantile levels (e.g., [0.1, 0.5, 0.9])

        Returns:
            ForecastResult with predictions and provenance

        Raises:
            RuntimeError: If prediction fails
            ValueError: If horizon exceeds model limits
        """
        pass

    @abstractmethod
    def get_model_signature(self) -> str:
        """Return unique signature for this model configuration.

        Used for provenance tracking to identify the exact model
        configuration used for a forecast.

        Returns:
            Unique signature string (e.g., "chronos-base-cuda-v1.0")
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory.

        Returns:
            True if model has been loaded, False otherwise
        """
        return self._model is not None

    def unload_model(self) -> None:
        """Unload model from memory to free resources.

        Useful for managing memory when using multiple large models.
        """
        self._model = None

    def _require_loaded(self, operation: str) -> None:
        """Ensure adapter model is already loaded by caller."""
        if self.is_loaded:
            return
        from tsagentkit.contracts import EModelNotLoaded

        raise EModelNotLoaded(
            f"Cannot call {operation} on adapter '{self.config.model_name}' before model is loaded.",
            context={
                "adapter_name": self.config.model_name,
                "operation": operation,
                "model_size": self.config.model_size,
                "device": self._device,
            },
        )

    def _validate_dataset(self, dataset: TSDataset) -> None:
        """Validate dataset compatibility.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If dataset is incompatible
        """
        # Check minimum length requirements
        min_length = 1  # Most TSFMs need at least some context

        for uid in dataset.series_ids:
            series = dataset.get_series(uid)
            if len(series) < min_length:
                raise ValueError(
                    f"Series '{uid}' has only {len(series)} observations. "
                    f"Minimum required: {min_length}"
                )

    def _create_provenance(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> Provenance:
        """Create provenance for forecast.

        Args:
            dataset: Input dataset
            horizon: Forecast horizon
            quantiles: Quantile levels used

        Returns:
            Provenance instance
        """
        # Create data signature from dataset
        import hashlib
        from datetime import datetime

        from tsagentkit.contracts import Provenance

        data_hash = hashlib.sha256(
            str(dataset.n_series).encode() +
            str(dataset.n_observations).encode() +
            dataset.date_range[0].isoformat().encode()
        ).hexdigest()[:16]

        return Provenance(
            run_id=f"tsfm_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(UTC).isoformat(),
            data_signature=data_hash,
            task_signature=f"horizon={horizon}",
            plan_signature=self.get_model_signature(),
            model_signature=self.get_model_signature(),
            metadata={
                "adapter": self.__class__.__name__,
                "device": self._device,
                "quantiles": quantiles,
            },
        )

    def _batch_iterator(
        self,
        data: list[Any],
        batch_size: int,
    ):
        """Iterate over data in batches.

        Args:
            data: List of data items
            batch_size: Batch size

        Yields:
            Batches of data
        """
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def _compute_quantiles_from_samples(
        self,
        samples: Any,  # numpy array or tensor
        quantiles: list[float],
    ) -> dict[float, float]:
        """Compute quantiles from sampled predictions.

        Args:
            samples: Array of samples with shape (n_samples, horizon)
            quantiles: Quantile levels to compute

        Returns:
            Dictionary mapping quantile levels to values
        """
        import numpy as np

        results = {}
        for q in quantiles:
            results[q] = float(np.quantile(samples, q, axis=0))
        return results

    def _handle_missing_values(
        self, values: pd.Series | np.ndarray
    ) -> pd.Series | np.ndarray:
        """Fill missing values using linear interpolation.

        Args:
            values: Series or array that may contain NaNs

        Returns:
            Values with NaNs filled
        """
        is_array = isinstance(values, np.ndarray)
        s = pd.Series(values).astype(float)
        s = s.interpolate(method="linear", limit_direction="both")
        if s.isna().any():
            fill_val = 0.0 if pd.isna(s.mean()) else s.mean()
            s = s.fillna(fill_val)
        return s.values if is_array else s

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check if required dependencies are installed.

        Raises:
            ImportError: If dependencies are missing
        """
        # Base class checks for torch only
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TSFM adapters. "
                "Install with: pip install torch"
            ) from e
        # Call adapter-specific dependency check
        cls._check_dependencies_impl()

    @classmethod
    @abstractmethod
    def _check_dependencies_impl(cls) -> None:
        """Check adapter-specific dependencies.

        Subclasses override this to check for their specific dependencies.

        Raises:
            ImportError: If adapter-specific dependencies are missing
        """
        pass

    @classmethod
    def capability(cls, adapter_name: str) -> AdapterCapabilitySpec:
        """Return static capability metadata for this adapter class.

        Args:
            adapter_name: Name of the adapter

        Returns:
            AdapterCapabilitySpec with adapter metadata
        """
        from tsagentkit.contracts import AdapterCapabilitySpec

        # Get adapter-specific capability details via template method
        caps = cls._get_capability_spec(adapter_name)

        # Merge with base defaults
        return AdapterCapabilitySpec(
            adapter_name=caps.get("adapter_name", adapter_name),
            provider=caps.get("provider"),
            available=None,
            availability_reason=None,
            is_zero_shot=caps.get("is_zero_shot", True),
            supports_quantiles=caps.get("supports_quantiles", True),
            supports_past_covariates=caps.get("supports_past_covariates", False),
            supports_future_covariates=caps.get("supports_future_covariates", False),
            supports_static_covariates=caps.get("supports_static_covariates", False),
            max_context_length=caps.get("max_context_length"),
            max_horizon=caps.get("max_horizon"),
            dependencies=caps.get("dependencies", ["torch"]),
            notes=caps.get("notes"),
        )

    @classmethod
    @abstractmethod
    def _get_capability_spec(cls, adapter_name: str) -> dict[str, Any]:
        """Return adapter-specific capability details.

        Subclasses override this to provide their specific capability metadata.

        Args:
            adapter_name: Name of the adapter

        Returns:
            Dictionary with capability fields
        """
        pass
