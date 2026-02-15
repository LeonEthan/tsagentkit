"""Functional pipeline for time-series forecasting.

Replaces the complex 903-line ForecastPipeline class with composable,
functional stages that can be mixed and matched.
"""

from tsagentkit.pipeline.runner import run_pipeline, forecast
from tsagentkit.pipeline.stages import STAGES, PipelineStage

__all__ = [
    "forecast",
    "run_pipeline",
    "STAGES",
    "PipelineStage",
]
