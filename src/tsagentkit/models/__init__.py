"""Models module for tsagentkit.

Provides model fitting, prediction, and TSFM (Time-Series Foundation Model)
adapters for various forecasting backends.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from typing import TYPE_CHECKING, Any

from tsagentkit.contracts import (
    EOOM,
    EModelFitFailed,
    EModelPredictFailed,
    ForecastResult,
    ModelArtifact,
    Provenance,
    TSAgentKitError,
)

# Import adapters submodules
from tsagentkit.models import adapters
from tsagentkit.models.baselines import fit_baseline, is_baseline_model, predict_baseline
from tsagentkit.models.sktime import SktimeModelBundle, fit_sktime, predict_sktime
from tsagentkit.utils import normalize_quantile_columns

if TYPE_CHECKING:
    from tsagentkit.contracts import AdapterCapabilitySpec, TaskSpec
    from tsagentkit.router import PlanSpec
    from tsagentkit.series import TSDataset


def _is_tsfm_model(model_name: str) -> bool:
    return model_name.lower().startswith("tsfm-")


def _is_sktime_model(model_name: str) -> bool:
    return model_name.lower().startswith("sktime-")


def _build_adapter_config(model_name: str, config: dict[str, Any]) -> adapters.AdapterConfig:
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


def _is_oom_error(error: Exception) -> bool:
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message or "cuda oom" in message or "cuda out of memory" in message


def _fit_model_name(
    model_name: str,
    dataset: TSDataset,
    plan: PlanSpec,
    covariates: Any | None = None,
) -> ModelArtifact:
    """Fit a model by name with baseline or TSFM dispatch."""
    try:
        config: dict[str, Any] = {
            "horizon": dataset.task_spec.horizon,
            "season_length": dataset.task_spec.season_length or 1,
            "quantiles": plan.quantiles,
        }

        if _is_tsfm_model(model_name):
            adapter_name = model_name.split("tsfm-", 1)[-1]
            adapter_config = _build_adapter_config(model_name, {})
            adapter = adapters.AdapterRegistry.create(adapter_name, adapter_config)
            adapter.fit(
                dataset=dataset,
                prediction_length=dataset.task_spec.horizon,
                quantiles=plan.quantiles,
            )
            return ModelArtifact(
                model=adapter,
                model_name=model_name,
                config=config,
                metadata={"adapter": adapter_name},
            )

        if is_baseline_model(model_name):
            return fit_baseline(model_name, dataset, config)

        if _is_sktime_model(model_name):
            return fit_sktime(model_name, dataset, plan, covariates=covariates)

        raise EModelFitFailed(
            f"Unknown model name: {model_name}",
            context={"model_name": model_name},
        )
    except TSAgentKitError:
        raise
    except Exception as exc:
        if _is_oom_error(exc):
            raise EOOM(
                f"Out-of-memory during fit for model '{model_name}'.",
                context={"model_name": model_name, "error": str(exc)},
            ) from exc
        raise EModelFitFailed(
            f"Model '{model_name}' fit failed: {exc}",
            context={
                "model_name": model_name,
                "error_type": type(exc).__name__,
            },
        ) from exc


def fit(
    dataset: TSDataset,
    plan: PlanSpec,
    on_fallback: Callable[[str, str, Exception], None] | None = None,
    covariates: Any | None = None,
) -> ModelArtifact:
    """Fit a model using the plan's fallback ladder."""
    from tsagentkit.router import execute_with_fallback

    def _fit(model_name: str, ds: TSDataset) -> ModelArtifact:
        return _fit_model_name(model_name, ds, plan, covariates=covariates)

    artifact, _ = execute_with_fallback(
        fit_func=_fit,
        dataset=dataset,
        plan=plan,
        on_fallback=on_fallback,
    )
    return artifact


def _basic_provenance(
    dataset: TSDataset,
    spec: TaskSpec,
    artifact: ModelArtifact,
) -> Provenance:
    from datetime import datetime

    from tsagentkit.utils import compute_data_signature

    return Provenance(
        run_id=f"model_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now(UTC).isoformat(),
        data_signature=compute_data_signature(dataset.df),
        task_signature=spec.model_hash(),
        plan_signature=artifact.signature,
        model_signature=artifact.signature,
        metadata={"provenance_incomplete": True},
    )


def predict(
    dataset: TSDataset,
    artifact: ModelArtifact,
    spec: TaskSpec,
    covariates: Any | None = None,
) -> ForecastResult:
    """Generate predictions for baseline or TSFM models."""
    try:
        if isinstance(artifact.model, adapters.TSFMAdapter):
            result = artifact.model.predict(
                dataset=dataset,
                horizon=spec.horizon,
                quantiles=artifact.config.get("quantiles"),
            )
            df = normalize_quantile_columns(result.df)
            if "model" not in df.columns:
                df = df.copy()
                df["model"] = artifact.model_name
            return ForecastResult(
                df=df,
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
            if "model" not in forecast_df.columns:
                forecast_df = forecast_df.copy()
                forecast_df["model"] = artifact.model_name
            provenance = _basic_provenance(dataset, spec, artifact)
            return ForecastResult(
                df=normalize_quantile_columns(forecast_df),
                provenance=provenance,
                model_name=artifact.model_name,
                horizon=spec.horizon,
            )

        if isinstance(artifact.model, SktimeModelBundle):
            return predict_sktime(
                dataset=dataset,
                artifact=artifact,
                spec=spec,
                covariates=covariates,
            )

        raise EModelPredictFailed(
            f"Unknown model type for prediction: {artifact.model_name}",
            context={"model_name": artifact.model_name},
        )
    except TSAgentKitError:
        raise
    except Exception as exc:
        if _is_oom_error(exc):
            raise EOOM(
                f"Out-of-memory during predict for model '{artifact.model_name}'.",
                context={"model_name": artifact.model_name, "error": str(exc)},
            ) from exc
        raise EModelPredictFailed(
            f"Model '{artifact.model_name}' predict failed: {exc}",
            context={
                "model_name": artifact.model_name,
                "error_type": type(exc).__name__,
            },
        ) from exc


def get_adapter_capability(name: str) -> AdapterCapabilitySpec:
    """Return capability metadata for a registered TSFM adapter."""
    return adapters.AdapterRegistry.get_capability(name)


def list_adapter_capabilities(
    names: list[str] | None = None,
) -> dict[str, AdapterCapabilitySpec]:
    """Return capability metadata for selected registered TSFM adapters."""
    return adapters.AdapterRegistry.list_capabilities(names=names)


__all__ = [
    "fit",
    "predict",
    "adapters",
    "get_adapter_capability",
    "list_adapter_capabilities",
]
