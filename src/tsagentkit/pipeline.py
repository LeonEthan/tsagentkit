"""Pipeline for time-series forecasting.

Consolidated pipeline with ModelCache optimization.
Core logic: validate → build dataset → fit TSFMs → ensemble → return
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.dataset import CovariateSet, TSDataset
from tsagentkit.core.errors import EContract, EInsufficient, ENoTSFM
from tsagentkit.core.results import ForecastResult
from tsagentkit.models.ensemble import ensemble_streaming, ensemble_with_quantiles
from tsagentkit.models.protocol import fit
from tsagentkit.models.protocol import predict as protocol_predict
from tsagentkit.models.registry import REGISTRY, ModelSpec, list_models

if TYPE_CHECKING:
    pass


# =============================================================================
# Validation
# =============================================================================


def validate(df: pd.DataFrame, config: ForecastConfig | None = None) -> pd.DataFrame:
    """Validate input data against the fixed panel contract.

    Args:
        df: Input DataFrame with time-series data
        config: Optional forecast config (unused, kept for API compatibility)

    Returns:
        Validated and normalized DataFrame
    """
    _ = config

    # Check required columns
    required = ["unique_id", "ds", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise EContract(
            f"Missing required columns: {missing}",
            context={"required": required, "found": list(df.columns)},
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise EContract("DataFrame is empty")

    # Check for nulls in key columns
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise EContract(
                f"Column '{col}' has {null_count} null values",
                fix_hint="Remove or fill null values in key columns",
            )

    return df.copy()


# =============================================================================
# Dataset Building
# =============================================================================


def build_dataset(
    df: pd.DataFrame,
    config: ForecastConfig,
    covariates: CovariateSet | None = None,
) -> TSDataset:
    """Build TSDataset from validated DataFrame.

    Args:
        df: Validated DataFrame
        config: Forecast configuration
        covariates: Optional covariate data

    Returns:
        TSDataset ready for forecasting
    """
    return TSDataset.from_dataframe(df, config, covariates)


# =============================================================================
# Planning
# =============================================================================

# Model speed ranking based on typical inference benchmarks
# Lower = faster
_MODEL_SPEED_RANK: dict[str, int] = {
    "timesfm": 1,  # Fastest - optimized inference
    "chronos": 2,  # Good speed with efficient batching
    "moirai": 3,  # Moderate speed
    "patchtst_fm": 4,  # Slower due to patching overhead
}

# Model accuracy ranking based on typical benchmark performance
# Lower = more accurate
_MODEL_ACCURACY_RANK: dict[str, int] = {
    "moirai": 1,  # Generally strongest on diverse datasets
    "chronos": 2,  # Strong zero-shot performance
    "timesfm": 3,  # Good on regular patterns
    "patchtst_fm": 4,  # Good but can be dataset-dependent
}


def _sort_by_speed(models: list[ModelSpec], dataset: TSDataset | None = None) -> list[ModelSpec]:
    """Sort models by inference speed priority."""

    def speed_key(spec: ModelSpec) -> int:
        return _MODEL_SPEED_RANK.get(spec.name, 99)

    return sorted(models, key=speed_key)


def _sort_by_accuracy(models: list[ModelSpec], dataset: TSDataset | None = None) -> list[ModelSpec]:
    """Sort models by accuracy priority."""

    def accuracy_key(spec: ModelSpec) -> int:
        return _MODEL_ACCURACY_RANK.get(spec.name, 99)

    return sorted(models, key=accuracy_key)


def make_plan(
    tsfm_only: bool = True,
    dataset: TSDataset | None = None,
    config: ForecastConfig | None = None,
) -> list[ModelSpec]:
    """Create execution plan with models for ensemble.

    Args:
        tsfm_only: If True, only include TSFM models
        dataset: Optional dataset for context-aware selection
        config: Optional config for model selection strategy

    Returns:
        List of model specifications to run
    """
    names = list_models(tsfm_only=tsfm_only)

    # TSFMs are required package dependencies. If this is empty, the registry
    # itself is misconfigured rather than dependencies being optional/missing.
    if tsfm_only and not names:
        raise ENoTSFM("No TSFM models registered")

    all_models = [REGISTRY[name] for name in names if name in REGISTRY]

    if config is None:
        return all_models

    # Apply model selection strategy
    if config.model_selection == "fast":
        all_models = _sort_by_speed(all_models, dataset)
    elif config.model_selection == "accurate":
        all_models = _sort_by_accuracy(all_models, dataset)

    # Apply max_models limit
    if config.max_models is not None:
        all_models = all_models[: config.max_models]

    return all_models


# =============================================================================
# Fitting and Prediction
# =============================================================================


def fit_all(models: list[ModelSpec], dataset: TSDataset, device: str | None = None) -> list[Any]:
    """Fit all models in the plan.

    Args:
        models: List of model specifications
        dataset: Time-series dataset
        device: Device to load TSFMs on ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        List of model artifacts (parallel to models)
    """
    artifacts = []
    for spec in models:
        try:
            artifact = fit(spec, dataset, device=device)
            artifacts.append(artifact)
        except Exception:
            artifacts.append(None)
    return artifacts


def predict_all(
    models: list[ModelSpec],
    artifacts: list[Any],
    dataset: TSDataset,
    h: int,
    quantiles: tuple[float, ...] | list[float] | None = None,
    batch_size: int = 32,
) -> list[pd.DataFrame]:
    """Generate predictions from all fitted models.

    Args:
        models: List of model specifications
        artifacts: List of model artifacts (parallel to models)
        dataset: Time-series dataset
        h: Forecast horizon
        quantiles: Optional quantile levels requested by the pipeline
        batch_size: Number of series to process in parallel

    Returns:
        List of forecast DataFrames
    """
    predictions = []
    for spec, artifact in zip(models, artifacts, strict=False):
        if artifact is None:
            continue
        try:
            pred = protocol_predict(
                spec, artifact, dataset, h, quantiles=quantiles, batch_size=batch_size
            )
            predictions.append(pred)
        except Exception:
            pass
    return predictions


def fit_all_parallel(
    models: list[ModelSpec],
    dataset: TSDataset,
    device: str | None = None,
    max_workers: int | None = None,
) -> list[Any]:
    """Fit models concurrently with controlled parallelism.

    Uses ThreadPoolExecutor for IO-bound model loading. Each model's fit
    operation runs in parallel, with failures handled gracefully.

    Args:
        models: List of model specifications
        dataset: Time-series dataset
        device: Device to load TSFMs on ('cuda', 'mps', 'cpu', or None for auto)
        max_workers: Maximum number of concurrent workers (None = auto)

    Returns:
        List of model artifacts (parallel to models), None for failed fits
    """

    def fit_one(spec: ModelSpec) -> Any:
        try:
            return fit(spec, dataset, device=device)
        except Exception:
            return None

    artifacts: list[Any] = [None] * len(models)

    # For single model or max_workers=1, fall back to sequential
    if len(models) <= 1 or max_workers == 1:
        return [fit_one(spec) for spec in models]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fit_one, spec): idx for idx, spec in enumerate(models)}
        for future in as_completed(futures):
            idx = futures[future]
            artifacts[idx] = future.result()

    return artifacts


def predict_all_parallel(
    models: list[ModelSpec],
    artifacts: list[Any],
    dataset: TSDataset,
    h: int,
    quantiles: tuple[float, ...] | list[float] | None = None,
    batch_size: int = 32,
    max_workers: int | None = None,
) -> list[pd.DataFrame]:
    """Generate predictions from all fitted models concurrently.

    Uses ThreadPoolExecutor for concurrent prediction. Note: This may
    increase memory usage as multiple models hold GPU memory simultaneously.
    Use with caution on GPU-bound systems.

    Args:
        models: List of model specifications
        artifacts: List of model artifacts (parallel to models)
        dataset: Time-series dataset
        h: Forecast horizon
        quantiles: Optional quantile levels requested by the pipeline
        batch_size: Number of series to process in parallel per model
        max_workers: Maximum number of concurrent workers (None = auto)

    Returns:
        List of forecast DataFrames
    """

    def predict_one(spec: ModelSpec, artifact: Any) -> pd.DataFrame | None:
        if artifact is None:
            return None
        try:
            return protocol_predict(
                spec, artifact, dataset, h, quantiles=quantiles, batch_size=batch_size
            )
        except Exception:
            return None

    # Filter out None artifacts and pair with models
    valid_pairs = [
        (spec, art) for spec, art in zip(models, artifacts, strict=False) if art is not None
    ]

    if not valid_pairs:
        return []

    # For single model or max_workers=1, fall back to sequential
    if len(valid_pairs) <= 1 or max_workers == 1:
        return [
            predict_one(spec, art)
            for spec, art in valid_pairs
            if predict_one(spec, art) is not None
        ]

    predictions: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(predict_one, spec, art): idx
            for idx, (spec, art) in enumerate(valid_pairs)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                predictions.append(result)

    return predictions


# =============================================================================
# Main Pipeline
# =============================================================================


def run_forecast(
    data: pd.DataFrame,
    config: ForecastConfig,
    covariates: CovariateSet | None = None,
) -> ForecastResult:
    """Run forecasting pipeline with config.

    This is the standard pipeline entry point for agent building.

    Args:
        data: Input DataFrame with time-series data
        config: Forecast configuration
        covariates: Optional covariate data

    Returns:
        ForecastResult with ensemble forecast
    """
    # Phase 1: Validate and build dataset
    df = validate(data, config)
    dataset = build_dataset(df, config, covariates)

    # Phase 2: Get TSFM models with selection strategy
    models = make_plan(tsfm_only=True, dataset=dataset, config=config)

    # Phase 3: Fit all models (uses ModelCache for TSFMs)
    if config.parallel_fit:
        artifacts = fit_all_parallel(
            models, dataset, device=config.device, max_workers=config.max_workers
        )
    else:
        artifacts = fit_all(models, dataset, device=config.device)

    # Check if all TSFMs failed to load (import/runtime errors)
    successful_fits = sum(1 for a in artifacts if a is not None)
    if successful_fits == 0:
        raise ENoTSFM(
            "No TSFM models could be loaded. All registered TSFMs failed due to import "
            "errors, missing dependencies, or runtime failures."
        )

    # Phase 4: Predict all
    if config.parallel_predict:
        predictions = predict_all_parallel(
            models,
            artifacts,
            dataset,
            config.h,
            quantiles=config.quantiles,
            batch_size=config.batch_size,
            max_workers=config.max_workers,
        )
    else:
        predictions = predict_all(
            models,
            artifacts,
            dataset,
            config.h,
            quantiles=config.quantiles,
            batch_size=config.batch_size,
        )

    # Check minimum models
    successful = len(predictions)
    if successful < config.min_tsfm:
        raise EInsufficient(
            f"Insufficient models succeeded: {successful} < {config.min_tsfm}",
        )

    # Phase 5: Ensemble
    # Auto-select streaming for large panels (>50k rows) to reduce memory usage
    if predictions and len(predictions[0]) > 50000:
        ensemble_df = ensemble_streaming(
            predictions,
            method=config.ensemble_method,
            quantiles=config.quantiles,
            chunk_size=5000,
        )
    else:
        ensemble_df = ensemble_with_quantiles(
            predictions,
            method=config.ensemble_method,
            quantiles=config.quantiles,
            quantile_mode=config.quantile_mode,
        )

    # Build result
    result = ForecastResult(
        df=ensemble_df,
        model_name=f"ensemble_{config.ensemble_method}",
        config=config,
    )

    return result


def forecast(
    data: pd.DataFrame,
    h: int,
    freq: str = "D",
    **kwargs,
) -> ForecastResult:
    """Zero-config TSFM ensemble forecast.

    The main user-facing API for quick forecasting.

    Args:
        data: Input DataFrame with columns [unique_id, ds, y]
        h: Forecast horizon (number of periods)
        freq: Time-series frequency ('D', 'H', 'M', etc.)
        **kwargs: Additional config options (quantiles, ensemble_method, etc.)

    Returns:
        ForecastResult with ensemble forecast

    Examples:
        >>> result = forecast(df, h=7)
        >>> print(result.df)
    """
    config = ForecastConfig(h=h, freq=freq, **kwargs)
    return run_forecast(data, config)


__all__ = [
    # Main entry points
    "forecast",
    "run_forecast",
    # Agent building (granular control)
    "validate",
    "build_dataset",
    "make_plan",
    "fit_all",
    "predict_all",
    # Parallel execution (Phase 1)
    "fit_all_parallel",
    "predict_all_parallel",
]
