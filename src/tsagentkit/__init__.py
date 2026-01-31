"""tsagentkit - Robust execution engine for time-series forecasting agents.

This library provides a strict, production-grade workflow skeleton for
external coding agents (LLMs/AI agents) performing time-series forecasting tasks.

Basic usage:
    >>> from tsagentkit import TaskSpec, validate_contract, run_forecast
    >>> spec = TaskSpec(horizon=7, freq="D")
    >>> result = run_forecast(data, spec)
"""

__version__ = "0.1.0"

# Import commonly used items from contracts
from tsagentkit.contracts import (
    TaskSpec,
    ValidationReport,
    ForecastResult,
    ModelArtifact,
    Provenance,
    RunArtifact,
    validate_contract,
    # Errors
    TSAgentKitError,
    ESplitRandomForbidden,
    ECovariateLeakage,
)
from tsagentkit.backtest import (
    BacktestReport,
    rolling_backtest,
    wape,
    smape,
    mase,
)
from tsagentkit.router import (
    Plan,
    make_plan,
    FallbackLadder,
    execute_with_fallback,
)
from tsagentkit.series import (
    TSDataset,
    SparsityProfile,
    SparsityClass,
    build_dataset,
)
from tsagentkit.serving import run_forecast

__all__ = [
    "__version__",
    # Core contracts
    "TaskSpec",
    "ValidationReport",
    "ForecastResult",
    "ModelArtifact",
    "Provenance",
    "RunArtifact",
    "validate_contract",
    # Key errors
    "TSAgentKitError",
    "ESplitRandomForbidden",
    "ECovariateLeakage",
    # Series
    "TSDataset",
    "SparsityProfile",
    "SparsityClass",
    "build_dataset",
    # Router
    "Plan",
    "make_plan",
    "FallbackLadder",
    "execute_with_fallback",
    # Backtest
    "BacktestReport",
    "rolling_backtest",
    "wape",
    "smape",
    "mase",
    # Serving
    "run_forecast",
]
