"""tsagentkit - Robust execution engine for time-series forecasting agents.

This library provides a strict, production-grade workflow skeleton for
external coding agents (LLMs/AI agents) performing time-series forecasting tasks.

Basic usage:
    >>> from tsagentkit import TaskSpec, validate_contract, run_qa, build_dataset, make_plan
    >>> spec = TaskSpec(h=7, freq="D")
    >>> report = validate_contract(data)
    >>> report.raise_if_errors()
    >>> qa = run_qa(data, spec, mode="quick")
    >>> dataset = build_dataset(data, spec)
    >>> plan, _route_decision = make_plan(dataset, spec)
"""

__version__ = "1.1.3"

# Import commonly used items from contracts
from tsagentkit.anomaly import detect_anomalies
from tsagentkit.backtest import (
    BacktestReport,
    MultiModelBacktestReport,
    SeriesModelRanking,
    multi_model_backtest,
    rolling_backtest,
)
from tsagentkit.calibration import apply_calibrator, fit_calibrator
from tsagentkit.contracts import (
    AdapterCapabilitySpec,
    DryRunResult,
    ECovariateLeakage,
    ESplitRandomForbidden,
    ForecastResult,
    ModelArtifact,
    PlanGraphSpec,
    PlanNodeSpec,
    Provenance,
    RunArtifact,
    TaskSpec,
    # Errors
    TSAgentKitError,
    TSFMPolicy,
    ValidationReport,
    validate_contract,
)
from tsagentkit.covariates import AlignedDataset, CovariateBundle, align_covariates

# API discovery (v1.1.1)
from tsagentkit.discovery import describe
from tsagentkit.eval import MetricFrame, ScoreSummary, evaluate_forecasts
from tsagentkit.models import (
    fit,
    fit_per_series,
    fit_predict_per_series,
    get_adapter_capability,
    list_adapter_capabilities,
    predict,
    predict_per_series,
)
from tsagentkit.qa import run_qa

# Quickstart (v1.1.1)
from tsagentkit.quickstart import diagnose, forecast

# Repair utility (v1.1.1)
from tsagentkit.repair import repair
from tsagentkit.router import (
    BucketConfig,
    BucketProfile,
    BucketStatistics,
    # Bucketing (v0.2)
    DataBucketer,
    FallbackLadder,
    PlanSpec,
    SeriesBucket,
    attach_plan_graph,
    build_plan_graph,
    compute_plan_signature,
    execute_with_fallback,
    get_candidate_models,
    inspect_tsfm_adapters,
    make_plan,
)
from tsagentkit.series import (
    SparsityClass,
    SparsityProfile,
    TSDataset,
    build_dataset,
)
from tsagentkit.serving import (
    MonitoringConfig,
    TSAgentSession,
    load_run_artifact,
    package_run,
    replay_forecast_from_artifact,
    run_forecast,
    save_run_artifact,
    validate_run_artifact_for_serving,
)

# Structured logging (v1.0)
from tsagentkit.serving.provenance import (
    StructuredLogger,
    format_event_json,
    log_event,
)

# v0.2 imports (optional - use directly from submodules)
# from tsagentkit.features import FeatureFactory, FeatureMatrix, compute_feature_hash
# from tsagentkit.monitoring import DriftDetector, StabilityMonitor, TriggerEvaluator

__all__ = [
    "__version__",
    # Core contracts
    "TaskSpec",
    "TSFMPolicy",
    "ValidationReport",
    "ForecastResult",
    "ModelArtifact",
    "PlanNodeSpec",
    "PlanGraphSpec",
    "AdapterCapabilitySpec",
    "Provenance",
    "RunArtifact",
    "DryRunResult",
    "validate_contract",
    "align_covariates",
    "CovariateBundle",
    "AlignedDataset",
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
    # Calibration + Anomaly + Eval
    "fit_calibrator",
    "apply_calibrator",
    "detect_anomalies",
    "evaluate_forecasts",
    "MetricFrame",
    "ScoreSummary",
    "fit",
    "predict",
    "fit_per_series",
    "predict_per_series",
    "fit_predict_per_series",
    "get_adapter_capability",
    "list_adapter_capabilities",
    # Router
    "PlanSpec",
    "build_plan_graph",
    "attach_plan_graph",
    "compute_plan_signature",
    "get_candidate_models",
    "make_plan",
    "inspect_tsfm_adapters",
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
    "MultiModelBacktestReport",
    "SeriesModelRanking",
    "rolling_backtest",
    "multi_model_backtest",
    # Serving
    "run_forecast",
    "TSAgentSession",
    "package_run",
    "save_run_artifact",
    "load_run_artifact",
    "validate_run_artifact_for_serving",
    "replay_forecast_from_artifact",
    "MonitoringConfig",
    # Structured Logging (v1.0)
    "log_event",
    "format_event_json",
    "StructuredLogger",
    # Repair utility (v1.1.1)
    "repair",
    # Quickstart (v1.1.1)
    "forecast",
    "diagnose",
    # API discovery (v1.1.1)
    "describe",
]
