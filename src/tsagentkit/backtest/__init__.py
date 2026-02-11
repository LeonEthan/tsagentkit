"""Backtest module for tsagentkit.

Provides rolling window backtesting and report structures. Legacy metric
helpers are still exported for compatibility but are deprecated; prefer
``tsagentkit.eval.evaluate_forecasts`` for new evaluation flows.
"""

from .engine import cross_validation_split, rolling_backtest
from .metrics import (
    compute_all_metrics,
    compute_metrics_by_series,
    mae,
    mase,
    pinball_loss,
    rmse,
    smape,
    wape,
    wql,
)
from .multi_model import (
    MultiModelBacktestReport,
    SeriesModelRanking,
    multi_model_backtest,
)
from .report import (
    BacktestReport,
    SegmentMetrics,
    SeriesMetrics,
    TemporalMetrics,
    WindowResult,
)

__all__ = [
    # Engine
    "rolling_backtest",
    "cross_validation_split",
    # Multi-model
    "multi_model_backtest",
    "MultiModelBacktestReport",
    "SeriesModelRanking",
    # Report
    "BacktestReport",
    "WindowResult",
    "SeriesMetrics",
    "SegmentMetrics",
    "TemporalMetrics",
    # Metrics
    "wape",
    "smape",
    "mase",
    "mae",
    "rmse",
    "pinball_loss",
    "wql",
    "compute_all_metrics",
    "compute_metrics_by_series",
]
