"""Serving module for tsagentkit.

Provides batch inference orchestration and artifact packaging.
"""

from tsagentkit.contracts import DryRunResult, RunArtifact

from .lifecycle import (
    load_run_artifact,
    replay_forecast_from_artifact,
    save_run_artifact,
    validate_run_artifact_for_serving,
)
from .model_pool import ModelPool, ModelPoolConfig
from .orchestration import MonitoringConfig, TSAgentSession, run_forecast
from .packaging import package_run
from .provenance import (
    StructuredLogger,
    compute_config_signature,
    compute_data_signature,
    create_provenance,
    format_event_json,
    log_event,
)
from .tsfm_cache import TSFMModelCache, clear_tsfm_cache, get_tsfm_model

__all__ = [
    # Orchestration
    "run_forecast",
    "TSAgentSession",
    "MonitoringConfig",
    "DryRunResult",
    # Packaging
    "package_run",
    "save_run_artifact",
    "load_run_artifact",
    "validate_run_artifact_for_serving",
    "replay_forecast_from_artifact",
    "RunArtifact",
    # Provenance
    "compute_data_signature",
    "compute_config_signature",
    "create_provenance",
    # Structured Logging
    "log_event",
    "format_event_json",
    "StructuredLogger",
    # TSFM Cache
    "TSFMModelCache",
    "get_tsfm_model",
    "clear_tsfm_cache",
    # Session ModelPool (Phase 1)
    "ModelPool",
    "ModelPoolConfig",
]
