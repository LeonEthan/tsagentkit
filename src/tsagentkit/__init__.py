"""tsagentkit - Robust execution engine for time-series forecasting agents.

This library provides a strict, production-grade workflow skeleton for
external coding agents (LLMs/AI agents) performing time-series forecasting tasks.

Basic usage:
    >>> from tsagentkit import TaskSpec, validate_contract, run_forecast
    >>> spec = TaskSpec(horizon=7, freq="D")
    >>> result = run_forecast(data, spec)
"""

__version__ = "0.2.0"

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
    # Bucketing (v0.2)
    DataBucketer,
    BucketConfig,
    BucketProfile,
    BucketStatistics,
    SeriesBucket,
)
from tsagentkit.series import (
    TSDataset,
    SparsityProfile,
    SparsityClass,
    build_dataset,
)
from tsagentkit.qa import run_qa
from tsagentkit.serving import MonitoringConfig, run_forecast

# v0.2 imports (optional - use directly from submodules)
# from tsagentkit.features import FeatureFactory, FeatureMatrix, compute_feature_hash
# from tsagentkit.monitoring import DriftDetector, StabilityMonitor, TriggerEvaluator

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
    # QA
    "run_qa",
    # Router
    "Plan",
    "make_plan",
    "FallbackLadder",
    "execute_with_fallback",
    # Router Bucketing (v0.2)
    "DataBucketer",
    "BucketConfig",
    "BucketProfile",
    "BucketStatistics",
    "SeriesBucket",
    # Backtest
    "BacktestReport",
    "rolling_backtest",
    "wape",
    "smape",
    "mase",
    # Serving
    "run_forecast",
    "MonitoringConfig",
]
