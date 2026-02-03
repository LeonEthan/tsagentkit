"""Pydantic specs for tsagentkit contracts and configuration.

These models are the JSON-serializable configuration and artifact contracts
used by agents and orchestration layers. They mirror docs/PRD.md Appendix B.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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
AnomalyMethod = Literal["interval_breach", "conformal_interval", "mad_residual"]
SeasonalityMethod = Literal["acf", "stl", "periodogram"]
CovariatePolicy = Literal["ignore", "known", "observed", "auto", "spec"]


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
    levels: List[int] = Field(default_factory=lambda: [80, 95])
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])


class CovariateSpec(BaseSpec):
    # Explicit typing strongly preferred for agent safety.
    roles: Dict[str, CovariateRole] = Field(default_factory=dict)
    missing_policy: MissingPolicy = "error"


# ---------------------------
# Task / execution specs
# ---------------------------

class BacktestSpec(BaseSpec):
    h: Optional[int] = Field(None, gt=0)
    n_windows: int = Field(5, gt=0)
    step: int = Field(1, gt=0)
    min_train_size: int = Field(56, gt=1)
    regularize_grid: bool = True


class TaskSpec(BaseSpec):
    # Forecast horizon
    h: int = Field(..., gt=0)

    # Frequency handling
    freq: str = Field(...)
    infer_freq: bool = True

    # Contracts
    panel_contract: PanelContract = Field(default_factory=PanelContract)
    forecast_contract: ForecastContract = Field(default_factory=ForecastContract)

    # Covariates
    covariates: Optional[CovariateSpec] = None
    covariate_policy: CovariatePolicy = "auto"

    # Backtest defaults (can be overridden by the caller)
    backtest: BacktestSpec = Field(default_factory=BacktestSpec)

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)

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
    def _apply_backtest_defaults(self) -> "TaskSpec":
        if self.backtest.h is None:
            object.__setattr__(self, "backtest", self.backtest.model_copy(update={"h": self.h}))
        return self

    @property
    def horizon(self) -> int:
        return self.h

    @property
    def quantiles(self) -> List[float]:
        return self.forecast_contract.quantiles

    @property
    def levels(self) -> List[int]:
        return self.forecast_contract.levels

    @property
    def season_length(self) -> int | None:
        return self._infer_season_length(self.freq)

    @staticmethod
    def _infer_season_length(freq: str) -> int | None:
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


class PlanSpec(BaseSpec):
    plan_name: str
    candidate_models: List[str] = Field(..., min_length=1)

    # Covariate usage rules
    use_static: bool = True
    use_past: bool = True
    use_future_known: bool = True

    # Training policy
    min_train_size: int = Field(56, gt=1)
    max_train_size: Optional[int] = None  # if set, truncate oldest points deterministically

    # Output policy
    interval_mode: IntervalMode = "level"
    levels: List[int] = Field(default_factory=lambda: [80, 95])
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Fallback policy
    allow_drop_covariates: bool = True
    allow_baseline: bool = True


class RouteDecision(BaseSpec):
    # Series statistics used in routing (computed deterministically)
    stats: Dict[str, Any] = Field(default_factory=dict)

    # Bucket tags
    buckets: List[str] = Field(default_factory=list)

    # Which plan template was selected
    selected_plan: PlanSpec

    # Human-readable deterministic reasons (safe for logs)
    reasons: List[str] = Field(default_factory=list)


class RouterConfig(BaseSpec):
    thresholds: RouterThresholds = Field(default_factory=RouterThresholds)

    # Mapping bucket -> plan template name, resolved by registry
    bucket_to_plan: Dict[str, str] = Field(default_factory=dict)

    # Default plan when no bucket matches
    default_plan: str = "default"


# ---------------------------
# Calibration + anomaly
# ---------------------------

class CalibratorSpec(BaseSpec):
    method: Literal["none", "conformal_interval"] = "conformal_interval"
    level: int = Field(99, ge=50, le=99)
    by: Optional[Literal["unique_id", "global"]] = "unique_id"


class AnomalySpec(BaseSpec):
    method: AnomalyMethod = "conformal_interval"
    level: int = Field(99, ge=50, le=99)
    score: Literal["margin", "normalized_margin", "zscore"] = "normalized_margin"


# ---------------------------
# Provenance artifacts (config-level, serializable)
# ---------------------------

class RunArtifactSpec(BaseSpec):
    run_id: str
    created_at: datetime

    task_spec: TaskSpec
    router_config: Optional[RouterConfig] = None
    route_decision: Optional[RouteDecision] = None

    # Identifiers / hashes for reproducibility (implementation-defined)
    data_signature: Optional[str] = None
    code_signature: Optional[str] = None

    # Output references (implementation-defined; typically file paths or object-store keys)
    outputs: Dict[str, str] = Field(default_factory=dict)


__all__ = [
    "AggregationMode",
    "AnomalyMethod",
    "AnomalySpec",
    "BacktestSpec",
    "BaseSpec",
    "CalibratorSpec",
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
    "TaskSpec",
]
