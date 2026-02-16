"""Pipeline for time-series forecasting.

Consolidated pipeline with ModelCache optimization.
Core logic: validate → build dataset → fit TSFMs → ensemble → return
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.dataset import CovariateSet, TSDataset
from tsagentkit.core.errors import EContract, EInsufficient, ENoTSFM
from tsagentkit.core.results import ForecastResult
from tsagentkit.models.ensemble import ensemble_with_quantiles
from tsagentkit.models.protocol import fit, predict
from tsagentkit.models.registry import REGISTRY, ModelSpec, list_available

if TYPE_CHECKING:
    pass


# =============================================================================
# Validation
# =============================================================================


def validate(df: pd.DataFrame, config: ForecastConfig | None = None) -> pd.DataFrame:
    """Validate input data and normalize column names.

    Args:
        df: Input DataFrame with time-series data
        config: Optional config for column name mapping

    Returns:
        Validated and normalized DataFrame
    """
    config = config or ForecastConfig(h=1)  # Dummy config for column names

    # Check required columns
    required = [config.id_col, config.time_col, config.target_col]
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

    # Normalize column names to standard
    df = df.copy()
    if config.id_col != "unique_id":
        df = df.rename(columns={config.id_col: "unique_id"})
    if config.time_col != "ds":
        df = df.rename(columns={config.time_col: "ds"})
    if config.target_col != "y":
        df = df.rename(columns={config.target_col: "y"})

    return df


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


def make_plan(
    tsfm_only: bool = True,
) -> list[ModelSpec]:
    """Create execution plan with models for ensemble.

    Args:
        tsfm_only: If True, only include TSFM models

    Returns:
        List of model specifications to run
    """
    available = list_available(tsfm_only=tsfm_only)

    if not available:
        raise ENoTSFM("No TSFM models available")

    return [REGISTRY[name] for name in available if name in REGISTRY]


# =============================================================================
# Fitting and Prediction
# =============================================================================


def fit_all(models: list[ModelSpec], dataset: TSDataset) -> list[Any]:
    """Fit all models in the plan.

    Args:
        models: List of model specifications
        dataset: Time-series dataset

    Returns:
        List of model artifacts (parallel to models)
    """
    artifacts = []
    for spec in models:
        try:
            artifact = fit(spec, dataset)
            artifacts.append(artifact)
        except Exception:
            artifacts.append(None)
    return artifacts


def predict_all(
    models: list[ModelSpec],
    artifacts: list[Any],
    dataset: TSDataset,
    h: int,
) -> list[pd.DataFrame]:
    """Generate predictions from all fitted models.

    Args:
        models: List of model specifications
        artifacts: List of model artifacts (parallel to models)
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        List of forecast DataFrames
    """
    predictions = []
    for spec, artifact in zip(models, artifacts):
        if artifact is None:
            continue
        try:
            pred = predict(spec, artifact, dataset, h)
            predictions.append(pred)
        except Exception:
            pass
    return predictions


# =============================================================================
# Ensemble
# =============================================================================


def ensemble(
    predictions: list[pd.DataFrame],
    method: str = "median",
    quantiles: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Compute ensemble forecast from multiple predictions.

    Args:
        predictions: List of forecast DataFrames
        method: Aggregation method ('median' or 'mean')
        quantiles: Optional quantile levels

    Returns:
        Ensemble forecast DataFrame
    """
    if not predictions:
        raise EInsufficient("No predictions to ensemble")

    if len(predictions) == 1:
        return predictions[0]

    return ensemble_with_quantiles(
        predictions,
        method=method,  # type: ignore[arg-type]
        quantiles=list(quantiles) if quantiles else None,
    )


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

    # Phase 2: Get TSFM models
    models = make_plan(tsfm_only=True)

    # Phase 3: Fit all models (uses ModelCache for TSFMs)
    artifacts = fit_all(models, dataset)

    # Phase 4: Predict all
    predictions = predict_all(models, artifacts, dataset, config.h)

    # Check minimum models
    successful = len(predictions)
    if successful < config.min_tsfm:
        raise EInsufficient(
            f"Insufficient models succeeded: {successful} < {config.min_tsfm}",
        )

    # Phase 5: Ensemble
    ensemble_df = ensemble(predictions, method=config.ensemble_method, quantiles=config.quantiles)

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
    "ensemble",
]
