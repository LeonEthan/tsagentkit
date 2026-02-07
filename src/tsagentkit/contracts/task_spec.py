"""Pydantic specs for tsagentkit contracts and configuration.

These models are the JSON-serializable configuration and artifact contracts
used by agents and orchestration layers. They mirror docs/PRD.md Appendix B.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------
# Common
# ---------------------------


class BaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CovariateRole = Literal["static", "past", "future_known"]
AggregationMode = Literal["reject", "sum", "mean", "median", "last"]
MissingPolicy = Literal["error", "ffill", "bfill", "zero", "mean"]
IntervalMode = Literal["level", "quantiles"]
AnomalyMethod = Literal["interval_breach", "conformal", "mad_residual"]
SeasonalityMethod = Literal["acf", "stl", "periodogram"]
CovariatePolicy = Literal["ignore", "known", "observed", "auto", "spec"]
TSFMMode = Literal["preferred", "required", "disabled"]
PlanNodeKind = Literal[
    "validate",
    "qa",
    "align_covariates",
    "build_dataset",
    "make_plan",
    "backtest",
    "fit",
    "predict",
    "package",
    "custom",
]


# ---------------------------
# Data contracts (column-level)
# ---------------------------


class PanelContract(BaseSpec):
    unique_id_col: str = "unique_id"
    ds_col: str = "ds"
    y_col: str = "y"
    aggregation: AggregationMode = "reject"


class ForecastContract(BaseSpec):
    long_format: bool = True
    model_col: str = "model"
    yhat_col: str = "yhat"
    cutoff_col: str = "cutoff"  # required for CV output
    interval_mode: IntervalMode = "level"
    levels: list[int] = Field(default_factory=lambda: [80, 95])
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])


class CovariateSpec(BaseSpec):
    # Explicit typing strongly preferred for agent safety.
    roles: dict[str, CovariateRole] = Field(default_factory=dict)
    missing_policy: MissingPolicy = "error"


class TSFMPolicy(BaseSpec):
    """Policy controlling TSFM availability requirements for routing."""

    mode: TSFMMode = "required"
    adapters: list[str] = Field(default_factory=lambda: ["chronos", "moirai", "timesfm"])
    allow_non_tsfm_fallback: bool | None = None

    @model_validator(mode="after")
    def _normalize_defaults(self) -> TSFMPolicy:
        if not self.adapters:
            raise ValueError("tsfm_policy.adapters must include at least one adapter.")
        if self.allow_non_tsfm_fallback is None:
            object.__setattr__(self, "allow_non_tsfm_fallback", self.mode != "required")
        return self


# ---------------------------
# Task / execution specs
# ---------------------------


class BacktestSpec(BaseSpec):
    h: int | None = Field(None, gt=0)
    n_windows: int = Field(5, gt=0)
    step: int = Field(1, gt=0)
    min_train_size: int = Field(56, gt=1)
    regularize_grid: bool = True


class TaskSpec(BaseSpec):
    # Forecast horizon
    h: int = Field(..., gt=0)

    # Frequency handling
    freq: str | None = None
    infer_freq: bool = True

    # Contracts
    panel_contract: PanelContract = Field(default_factory=PanelContract)
    forecast_contract: ForecastContract = Field(default_factory=ForecastContract)

    # Covariates
    covariates: CovariateSpec | None = None
    covariate_policy: CovariatePolicy = "auto"
    tsfm_policy: TSFMPolicy = Field(default_factory=TSFMPolicy)

    # Backtest defaults (can be overridden by the caller)
    backtest: BacktestSpec = Field(default_factory=BacktestSpec)

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)

        # Backward-compat aliases for TSFM policy
        if "tsfm" in payload and "tsfm_policy" not in payload:
            payload["tsfm_policy"] = payload.pop("tsfm")

        tsfm_policy = payload.get("tsfm_policy")
        if isinstance(tsfm_policy, TSFMPolicy):
            tsfm_policy = tsfm_policy.model_dump()

        if "require_tsfm" in payload:
            require_tsfm = bool(payload.pop("require_tsfm"))
            if not isinstance(tsfm_policy, dict):
                tsfm_policy = {}
            tsfm_policy = dict(tsfm_policy)
            tsfm_policy["mode"] = (
                "required"
                if require_tsfm
                else tsfm_policy.get(
                    "mode",
                    "preferred",
                )
            )

        if "tsfm_preference" in payload:
            preference = payload.pop("tsfm_preference")
            if not isinstance(tsfm_policy, dict):
                tsfm_policy = {}
            tsfm_policy = dict(tsfm_policy)
            if "adapters" not in tsfm_policy:
                tsfm_policy["adapters"] = preference

        if tsfm_policy is not None:
            payload["tsfm_policy"] = tsfm_policy

        # Backward-compat aliases
        if "horizon" in payload and "h" not in payload:
            payload["h"] = payload.pop("horizon")
        if "rolling_step" in payload:
            backtest = payload.get("backtest", {})
            if isinstance(backtest, BacktestSpec):
                backtest = backtest.model_dump()
            if isinstance(backtest, dict):
                backtest = dict(backtest)
                if "step" not in backtest:
                    backtest["step"] = payload.pop("rolling_step")
                payload["backtest"] = backtest

        # Legacy quantiles/levels mapping to forecast_contract
        if "quantiles" in payload or "levels" in payload:
            fc = payload.get("forecast_contract", {})
            if isinstance(fc, ForecastContract):
                fc = fc.model_dump()
            if isinstance(fc, dict):
                fc = dict(fc)
                if "quantiles" in payload:
                    fc["quantiles"] = payload.pop("quantiles")
                if "levels" in payload:
                    fc["levels"] = payload.pop("levels")
                payload["forecast_contract"] = fc

        return payload

    @model_validator(mode="after")
    def _apply_backtest_defaults(self) -> TaskSpec:
        if self.backtest.h is None:
            object.__setattr__(self, "backtest", self.backtest.model_copy(update={"h": self.h}))
        return self

    @property
    def horizon(self) -> int:
        return self.h

    @property
    def quantiles(self) -> list[float]:
        return self.forecast_contract.quantiles

    @property
    def levels(self) -> list[int]:
        return self.forecast_contract.levels

    @property
    def season_length(self) -> int | None:
        return self._infer_season_length(self.freq)

    @staticmethod
    def _infer_season_length(freq: str | None) -> int | None:
        if not freq:
            return None
        freq_map: dict[str, int] = {
            "D": 7,
            "B": 5,
            "H": 24,
            "T": 60,
            "min": 60,
            "M": 12,
            "MS": 12,
            "Q": 4,
            "QS": 4,
            "W": 52,
        }
        base_freq = freq.lstrip("0123456789")
        return freq_map.get(base_freq)

    def model_hash(self) -> str:
        import hashlib
        import json

        data = self.model_dump(exclude_none=True)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# ---------------------------
# Router / planning
# ---------------------------


class RouterThresholds(BaseSpec):
    min_train_size: int = Field(56, gt=1)
    max_missing_ratio: float = Field(0.15, ge=0.0, le=1.0)

    # Intermittency classification (heuristic, deterministic)
    max_intermittency_adi: float = Field(1.32, gt=0.0)
    max_intermittency_cv2: float = Field(0.49, ge=0.0)

    # Seasonality
    seasonality_method: SeasonalityMethod = "acf"
    min_seasonality_conf: float = Field(0.70, ge=0.0, le=1.0)

    # Practical routing guardrails
    max_series_count_for_tsfm: int = Field(20000, gt=0)
    max_points_per_series_for_tsfm: int = Field(5000, gt=0)

    # Configurable candidate model lists per bucket
    intermittent_candidates: list[str] = Field(default_factory=lambda: ["Croston", "Naive"])
    short_history_candidates: list[str] = Field(
        default_factory=lambda: ["HistoricAverage", "Naive"]
    )
    default_candidates: list[str] = Field(
        default_factory=lambda: ["SeasonalNaive", "HistoricAverage", "Naive"]
    )


class PlanNodeSpec(BaseSpec):
    node_id: str
    kind: PlanNodeKind
    depends_on: list[str] = Field(default_factory=list)
    model_name: str | None = None
    group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanGraphSpec(BaseSpec):
    plan_name: str
    nodes: list[PlanNodeSpec] = Field(..., min_length=1)
    entrypoints: list[str] = Field(default_factory=list)
    terminal_nodes: list[str] = Field(default_factory=list)
    version: int = 1

    @model_validator(mode="after")
    def _validate_graph(self) -> PlanGraphSpec:
        node_ids = {node.node_id for node in self.nodes}

        if not self.entrypoints:
            inferred = [node.node_id for node in self.nodes if not node.depends_on]
            object.__setattr__(self, "entrypoints", inferred)

        if not self.terminal_nodes:
            dependent_nodes = {dep for node in self.nodes for dep in node.depends_on}
            inferred = [node.node_id for node in self.nodes if node.node_id not in dependent_nodes]
            object.__setattr__(self, "terminal_nodes", inferred)

        missing_entrypoints = [node_id for node_id in self.entrypoints if node_id not in node_ids]
        missing_terminals = [node_id for node_id in self.terminal_nodes if node_id not in node_ids]
        if missing_entrypoints or missing_terminals:
            raise ValueError(
                "PlanGraph references unknown nodes.",
            )

        for node in self.nodes:
            missing_deps = [dep for dep in node.depends_on if dep not in node_ids]
            if missing_deps:
                raise ValueError(
                    f"PlanNode '{node.node_id}' depends on unknown nodes: {missing_deps}",
                )
        return self


class PlanSpec(BaseSpec):
    plan_name: str
    candidate_models: list[str] = Field(..., min_length=1)

    # Covariate usage rules
    use_static: bool = True
    use_past: bool = True
    use_future_known: bool = True

    # Training policy
    min_train_size: int = Field(56, gt=1)
    max_train_size: int | None = None  # if set, truncate oldest points deterministically

    # Output policy
    interval_mode: IntervalMode = "level"
    levels: list[int] = Field(default_factory=lambda: [80, 95])
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Fallback policy
    allow_drop_covariates: bool = True
    allow_baseline: bool = True
    graph: PlanGraphSpec | None = None


class AdapterCapabilitySpec(BaseSpec):
    adapter_name: str
    provider: str | None = None
    available: bool | None = None
    availability_reason: str | None = None
    is_zero_shot: bool = True
    supports_quantiles: bool = True
    supports_past_covariates: bool = False
    supports_future_covariates: bool = False
    supports_static_covariates: bool = False
    max_context_length: int | None = None
    max_horizon: int | None = None
    dependencies: list[str] = Field(default_factory=list)
    notes: str | None = None


class RouteDecision(BaseSpec):
    # Series statistics used in routing (computed deterministically)
    stats: dict[str, Any] = Field(default_factory=dict)

    # Bucket tags
    buckets: list[str] = Field(default_factory=list)

    # Which plan template was selected
    selected_plan: PlanSpec

    # Human-readable deterministic reasons (safe for logs)
    reasons: list[str] = Field(default_factory=list)


class RouterConfig(BaseSpec):
    thresholds: RouterThresholds = Field(default_factory=RouterThresholds)

    # Mapping bucket -> plan template name, resolved by registry
    bucket_to_plan: dict[str, str] = Field(default_factory=dict)

    # Default plan when no bucket matches
    default_plan: str = "default"


# ---------------------------
# Calibration + anomaly
# ---------------------------


class CalibratorSpec(BaseSpec):
    method: Literal["none", "conformal"] = "conformal"
    level: int = Field(99, ge=50, le=99)
    by: Literal["unique_id", "global"] = "unique_id"


class AnomalySpec(BaseSpec):
    method: AnomalyMethod = "conformal"
    level: int = Field(99, ge=50, le=99)
    score: Literal["margin", "normalized_margin", "zscore"] = "normalized_margin"


# ---------------------------
# Provenance artifacts (config-level, serializable)
# ---------------------------


class RunArtifactSpec(BaseSpec):
    run_id: str
    created_at: datetime

    task_spec: TaskSpec
    router_config: RouterConfig | None = None
    route_decision: RouteDecision | None = None

    # Identifiers / hashes for reproducibility (implementation-defined)
    data_signature: str | None = None
    code_signature: str | None = None

    # Output references (implementation-defined; typically file paths or object-store keys)
    outputs: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "AggregationMode",
    "AnomalyMethod",
    "AnomalySpec",
    "BacktestSpec",
    "BaseSpec",
    "CalibratorSpec",
    "PlanNodeKind",
    "PlanNodeSpec",
    "PlanGraphSpec",
    "AdapterCapabilitySpec",
    "CovariatePolicy",
    "CovariateRole",
    "CovariateSpec",
    "ForecastContract",
    "IntervalMode",
    "MissingPolicy",
    "PanelContract",
    "PlanSpec",
    "RouteDecision",
    "RouterConfig",
    "RouterThresholds",
    "RunArtifactSpec",
    "SeasonalityMethod",
    "TSFMMode",
    "TSFMPolicy",
    "TaskSpec",
]
