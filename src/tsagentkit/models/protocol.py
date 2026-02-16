"""Pure function protocol for model fitting and prediction.

This module defines the protocol interface for models using pure functions
rather than class inheritance. This is simpler, more flexible, and easier
to understand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.models.cache import ModelCache
from tsagentkit.models.registry import ModelSpec

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


ModelArtifact = Any  # Adapter decides what to store


def load(spec: ModelSpec) -> ModelArtifact:
    """Load a model.

    For TSFMs, this loads the pretrained model weights.
    For statistical models, this is a no-op (stateless).

    Args:
        spec: Model specification

    Returns:
        Model artifact (type depends on adapter)
    """
    import importlib

    module = importlib.import_module(spec.adapter_path)
    load_fn = getattr(module, "load")
    return load_fn(**spec.config_fields)


def fit(spec: ModelSpec, dataset: TSDataset) -> ModelArtifact:
    """Fit a model to the dataset.

    For TSFMs, this returns the loaded model (pre-trained, no fitting needed).
    For statistical models, this fits the model to the data.

    Args:
        spec: Model specification
        dataset: Time-series dataset

    Returns:
        Model artifact for prediction
    """
    if spec.is_tsfm:
        # TSFMs are pre-trained, just load from cache
        return ModelCache.get(spec)
    else:
        # Statistical models need fitting
        import importlib

        module = importlib.import_module(spec.adapter_path)
        fit_fn = getattr(module, "fit")
        return fit_fn(dataset)


def predict(
    spec: ModelSpec,
    artifact: ModelArtifact,
    dataset: TSDataset,
    h: int,
) -> pd.DataFrame:
    """Generate predictions from a fitted model.

    Args:
        spec: Model specification
        artifact: Model artifact from fit()
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import importlib

    module = importlib.import_module(spec.adapter_path)
    predict_fn = getattr(module, "predict")
    return predict_fn(artifact, dataset, h)


def predict_all(
    specs: list[ModelSpec],
    artifacts: list[ModelArtifact],
    dataset: TSDataset,
    h: int,
) -> list[pd.DataFrame]:
    """Generate predictions from multiple models.

    Args:
        specs: List of model specifications
        artifacts: List of model artifacts (parallel to specs)
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        List of forecast DataFrames
    """
    predictions = []
    for spec, artifact in zip(specs, artifacts):
        try:
            pred = predict(spec, artifact, dataset, h)
            predictions.append(pred)
        except Exception:
            # Skip failed predictions
            pass
    return predictions


__all__ = [
    "ModelArtifact",
    "load",
    "fit",
    "predict",
    "predict_all",
]
