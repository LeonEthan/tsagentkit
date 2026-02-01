"""Models module for tsagentkit.

Provides model fitting, prediction, and TSFM (Time-Series Foundation Model)
adapters for various forecasting backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.contracts import ModelArtifact
from tsagentkit.models.baselines import fit_baseline, is_baseline_model, predict_baseline
from tsagentkit.utils import normalize_quantile_columns

# Import adapters submodules
from tsagentkit.models import adapters

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset
    from tsagentkit.models.adapters import TSFMAdapter


def _is_tsfm_model(model_name: str) -> bool:
    return model_name.lower().startswith("tsfm-")


def _build_adapter_config(model_name: str, config: dict[str, Any]) -> "adapters.AdapterConfig":
    adapter_name = model_name.split("tsfm-", 1)[-1]
    return adapters.AdapterConfig(
        model_name=adapter_name,
        model_size=config.get("model_size", "base"),
        device=config.get("device"),
        cache_dir=config.get("cache_dir"),
        batch_size=config.get("batch_size", 32),
        prediction_batch_size=config.get("prediction_batch_size", 100),
        quantile_method=config.get("quantile_method", "sample"),
        num_samples=config.get("num_samples", 100),
        max_context_length=config.get("max_context_length"),
    )


def fit(model_name: str, dataset: TSDataset, config: dict[str, Any]) -> ModelArtifact:
    """Fit a model with baseline or TSFM dispatch."""
    if _is_tsfm_model(model_name):
        adapter_name = model_name.split("tsfm-", 1)[-1]
        adapter_config = _build_adapter_config(model_name, config)
        adapter = adapters.AdapterRegistry.create(adapter_name, adapter_config)
        adapter.fit(
            dataset=dataset,
            prediction_length=config.get("horizon", dataset.task_spec.horizon),
            quantiles=config.get("quantiles"),
        )
        return ModelArtifact(
            model=adapter,
            model_name=model_name,
            config=config,
            metadata={"adapter": adapter_name},
        )

    if is_baseline_model(model_name):
        return fit_baseline(model_name, dataset, config)

    raise ValueError(f"Unknown model name: {model_name}")


def predict(
    model: ModelArtifact,
    dataset: TSDataset,
    horizon: int,
) -> pd.DataFrame:
    """Generate predictions for baseline or TSFM models."""
    if isinstance(model.model, adapters.TSFMAdapter):
        result = model.model.predict(
            dataset=dataset,
            horizon=horizon,
            quantiles=model.config.get("quantiles"),
        )
        return normalize_quantile_columns(result.df)

    if is_baseline_model(model.model_name):
        forecast_df = predict_baseline(
            model_artifact=model,
            dataset=dataset,
            horizon=horizon,
            quantiles=model.config.get("quantiles"),
        )
        return normalize_quantile_columns(forecast_df)

    raise ValueError(f"Unknown model type for prediction: {model.model_name}")


__all__ = ["fit", "predict", "adapters"]
