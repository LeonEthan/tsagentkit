"""Pipeline runner - functional composition for forecasting.

Replaces the 903-line ForecastPipeline class with simple functional composition.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd

from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.data import CovariateSet
from tsagentkit.core.results import RunResult

if TYPE_CHECKING:
    from tsagentkit.pipeline.stages import PipelineStage


def run_pipeline(
    data: pd.DataFrame,
    config: ForecastConfig,
    covariates: CovariateSet | None = None,
    stages: list[PipelineStage] | None = None,
) -> RunResult:
    """Run forecasting pipeline with specified stages.

    This is the core execution engine that composes stages functionally.
    Each stage receives the output of the previous stage.

    Args:
        data: Input DataFrame with time-series data
        config: Forecast configuration
        covariates: Optional covariate data
        stages: Custom pipeline stages (uses default if None)

    Returns:
        RunResult with forecast and metadata
    """
    from tsagentkit.pipeline.stages import (
        STAGES,
        backtest_stage,
        build_dataset_stage,
        fit_predict_stage,
        package_stage,
        plan_stage,
        qa_stage,
        validate_stage,
    )

    start_time = time.time()

    # Use default stages if not specified
    stages = stages or STAGES

    # Phase 1: Data preparation
    df = validate_stage(data, config)
    df = qa_stage(df, config)
    dataset = build_dataset_stage(df, config, covariates)

    # Phase 2: Planning
    dataset, candidates = plan_stage(dataset, config)

    # Phase 3: Optional backtest
    dataset, candidates, _ = backtest_stage(dataset, candidates, config)

    # Phase 4: Fit and predict with fallback
    forecast_result, fallbacks = fit_predict_stage(dataset, candidates, config)

    # Phase 5: Package
    duration_ms = (time.time() - start_time) * 1000
    result = package_stage(forecast_result, fallbacks, config, duration_ms)

    return result


def forecast(
    data: pd.DataFrame,
    h: int,
    freq: str = "D",
    **kwargs,
) -> RunResult:
    """Single-entry-point forecasting.

    The main user-facing API for tsagentkit. This is the unified gateway
    that replaces run_forecast(), TSAgentSession, and ForecastPipeline.

    Args:
        data: Input DataFrame with columns [unique_id, ds, y]
        h: Forecast horizon (number of periods)
        freq: Time-series frequency ('D', 'H', 'M', etc.)
        **kwargs: Additional config options (quantiles, mode, tsfm_mode, etc.)

    Returns:
        RunResult with forecast and metadata

    Examples:
        # Quick forecast
        result = forecast(df, h=7)
        print(result.forecast.df)

        # With configuration
        result = forecast(df, h=14, freq='H', mode='strict')

        # Using presets
        result = forecast(df, **ForecastConfig.quick(h=7).to_dict())
    """
    # Build config from args
    config = ForecastConfig(h=h, freq=freq, **kwargs)

    # Run pipeline
    return run_pipeline(data, config)


def quick_forecast(data: pd.DataFrame, h: int, freq: str = "D") -> RunResult:
    """Quick forecast preset - minimal backtest, allows fallback."""
    config = ForecastConfig.quick(h=h, freq=freq)
    return run_pipeline(data, config)


def strict_forecast(data: pd.DataFrame, h: int, freq: str = "D") -> RunResult:
    """Strict forecast preset - fail fast, no fallback."""
    config = ForecastConfig.strict(h=h, freq=freq)
    return run_pipeline(data, config)
