"""Pure function protocol for model fitting and prediction.

This module defines the protocol interface for models using pure functions
rather than class inheritance. This is simpler, more flexible, and easier
to understand.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pandas as pd

from tsagentkit.core.types import ModelArtifact
from tsagentkit.models.cache import ModelCache
from tsagentkit.models.registry import ModelSpec

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def fit(spec: ModelSpec, dataset: TSDataset, device: str | None = None) -> ModelArtifact:
    """Fit a model to the dataset.

    For TSFMs, this returns the loaded model (pre-trained, no fitting needed).
    For statistical models, this fits the model to the data.

    Args:
        spec: Model specification
        dataset: Time-series dataset
        device: Device to load TSFM on ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        Model artifact for prediction
    """
    if spec.is_tsfm:
        # TSFMs are pre-trained, just load from cache
        return ModelCache.get(spec, device=device)
    else:
        # Statistical models need fitting
        import importlib

        module = importlib.import_module(spec.adapter_path)
        fit_fn = module.fit
        return fit_fn(dataset)


def predict(
    spec: ModelSpec,
    artifact: ModelArtifact,
    dataset: TSDataset,
    h: int,
    quantiles: tuple[float, ...] | list[float] | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Generate predictions from a fitted model.

    Args:
        spec: Model specification
        artifact: Model artifact from fit()
        dataset: Time-series dataset
        h: Forecast horizon
        quantiles: Optional quantile levels requested by the pipeline
        batch_size: Number of series to process in parallel

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import importlib

    module = importlib.import_module(spec.adapter_path)
    predict_fn = module.predict

    try:
        signature = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        signature = None

    params = signature.parameters if signature else {}
    has_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

    supports_quantiles = "quantiles" in params or has_var_kw
    supports_batch_size = "batch_size" in params or has_var_kw

    kwargs: dict = {}
    if supports_quantiles:
        kwargs["quantiles"] = quantiles
    if supports_batch_size:
        kwargs["batch_size"] = batch_size

    return predict_fn(artifact, dataset, h, **kwargs)


def predict_all(
    specs: list[ModelSpec],
    artifacts: list[ModelArtifact],
    dataset: TSDataset,
    h: int,
    quantiles: tuple[float, ...] | list[float] | None = None,
    batch_size: int = 32,
) -> list[pd.DataFrame]:
    """Generate predictions from multiple models.

    Args:
        specs: List of model specifications
        artifacts: List of model artifacts (parallel to specs)
        dataset: Time-series dataset
        h: Forecast horizon
        quantiles: Optional quantile levels requested by the pipeline
        batch_size: Number of series to process in parallel

    Returns:
        List of forecast DataFrames
    """
    return [
        predict(spec, artifact, dataset, h, quantiles=quantiles, batch_size=batch_size)
        for spec, artifact in zip(specs, artifacts, strict=False)
    ]


__all__ = [
    "ModelArtifact",
    "fit",
    "predict",
    "predict_all",
]
