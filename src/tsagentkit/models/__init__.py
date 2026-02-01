"""Models module for tsagentkit.

Provides model fitting, prediction, and TSFM (Time-Series Foundation Model)
adapters for various forecasting backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from tsagentkit.contracts import ModelArtifact, ForecastResult, Provenance
from tsagentkit.models.baselines import fit_baseline, is_baseline_model, predict_baseline
from tsagentkit.utils import normalize_quantile_columns

# Import adapters submodules
from tsagentkit.models import adapters

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset
    from tsagentkit.models.adapters import TSFMAdapter
    from tsagentkit.router import Plan
    from tsagentkit.contracts import TaskSpec


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


def _fit_model_name(model_name: str, dataset: "TSDataset", config: dict[str, Any]) -> ModelArtifact:
    """Fit a model by name with baseline or TSFM dispatch."""
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


def fit(
    dataset: "TSDataset",
    plan: "Plan",
    on_fallback: Callable[[str, str, Exception], None] | None = None,
) -> ModelArtifact:
    """Fit a model using the plan's fallback ladder."""
    from tsagentkit.router import execute_with_fallback

    artifact, _ = execute_with_fallback(
        fit_func=_fit_model_name,
        dataset=dataset,
        plan=plan,
        on_fallback=on_fallback,
    )
    return artifact


def _basic_provenance(
    dataset: "TSDataset",
    spec: "TaskSpec",
    artifact: ModelArtifact,
) -> Provenance:
    from datetime import datetime, timezone

    from tsagentkit.serving.provenance import compute_data_signature

    return Provenance(
        run_id=f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_signature=compute_data_signature(dataset.df),
        task_signature=spec.model_hash(),
        plan_signature=artifact.signature,
        model_signature=artifact.signature,
        metadata={"provenance_incomplete": True},
    )


def predict(
    dataset: "TSDataset",
    artifact: ModelArtifact,
    spec: "TaskSpec",
) -> ForecastResult:
    """Generate predictions for baseline or TSFM models."""
    if isinstance(artifact.model, adapters.TSFMAdapter):
        result = artifact.model.predict(
            dataset=dataset,
            horizon=spec.horizon,
            quantiles=artifact.config.get("quantiles"),
        )
        return ForecastResult(
            df=normalize_quantile_columns(result.df),
            provenance=result.provenance,
            model_name=artifact.model_name,
            horizon=spec.horizon,
        )

    if is_baseline_model(artifact.model_name):
        forecast_df = predict_baseline(
            model_artifact=artifact,
            dataset=dataset,
            horizon=spec.horizon,
            quantiles=artifact.config.get("quantiles"),
        )
        provenance = _basic_provenance(dataset, spec, artifact)
        return ForecastResult(
            df=normalize_quantile_columns(forecast_df),
            provenance=provenance,
            model_name=artifact.model_name,
            horizon=spec.horizon,
        )

    raise ValueError(f"Unknown model type for prediction: {artifact.model_name}")


__all__ = ["fit", "predict", "adapters"]
