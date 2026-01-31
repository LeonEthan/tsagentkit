"""Backtest module for tsagentkit.

Provides rolling window backtesting and metrics calculation.
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
)
from .report import BacktestReport, SeriesMetrics, WindowResult

__all__ = [
    # Engine
    "rolling_backtest",
    "cross_validation_split",
    # Report
    "BacktestReport",
    "WindowResult",
    "SeriesMetrics",
    # Metrics
    "wape",
    "smape",
    "mase",
    "mae",
    "rmse",
    "pinball_loss",
    "compute_all_metrics",
    "compute_metrics_by_series",
]
