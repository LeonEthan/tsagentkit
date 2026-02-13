"""TaskSpec normalization and backward-compatibility handling.

This module handles normalization of TaskSpec payloads, including:
- Legacy field name migrations (tsfm -> tsfm_policy, horizon -> h, etc.)
- Backward-compatible handling of deprecated parameters
- Migration of standalone fields to nested contract objects
"""

from __future__ import annotations

from typing import Any

from tsagentkit.contracts.task_spec import BacktestSpec, ForecastContract, TSFMPolicy


def normalize_task_spec_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy TaskSpec field names and structures.

    Handles the following migrations:
    - tsfm -> tsfm_policy
    - require_tsfm -> tsfm_policy.mode
    - tsfm_preference -> tsfm_policy.adapters
    - horizon -> h
    - rolling_step -> backtest.step
    - quantiles/levels -> forecast_contract.quantiles/levels

    Args:
        payload: Raw TaskSpec payload dictionary

    Returns:
        Normalized payload dictionary
    """
    result = dict(payload)

    # Normalize TSFM policy fields
    result = _normalize_tsfm_policy(result)

    # Normalize backward-compat aliases
    result = _normalize_field_aliases(result)

    # Normalize legacy quantiles/levels to forecast_contract
    result = _normalize_forecast_contract(result)

    return result


def _normalize_tsfm_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize TSFM policy related fields.

    Handles:
    - tsfm -> tsfm_policy rename
    - require_tsfm -> tsfm_policy.mode
    - tsfm_preference -> tsfm_policy.adapters
    """
    # Backward-compat alias: tsfm -> tsfm_policy
    if "tsfm" in payload and "tsfm_policy" not in payload:
        payload["tsfm_policy"] = payload.pop("tsfm")

    tsfm_policy = payload.get("tsfm_policy")
    if isinstance(tsfm_policy, TSFMPolicy):
        tsfm_policy = tsfm_policy.model_dump()

    # Handle require_tsfm -> mode migration
    if "require_tsfm" in payload:
        require_tsfm = bool(payload.pop("require_tsfm"))
        if not isinstance(tsfm_policy, dict):
            tsfm_policy = {}
        tsfm_policy = dict(tsfm_policy)
        tsfm_policy["mode"] = (
            "required"
            if require_tsfm
            else tsfm_policy.get("mode", "preferred")
        )

    # Handle tsfm_preference -> adapters migration
    if "tsfm_preference" in payload:
        preference = payload.pop("tsfm_preference")
        if not isinstance(tsfm_policy, dict):
            tsfm_policy = {}
        tsfm_policy = dict(tsfm_policy)
        if "adapters" not in tsfm_policy:
            tsfm_policy["adapters"] = preference

    if tsfm_policy is not None:
        payload["tsfm_policy"] = tsfm_policy

    return payload


def _normalize_field_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize field name aliases.

    Handles:
    - horizon -> h
    - rolling_step -> backtest.step
    """
    # horizon -> h
    if "horizon" in payload and "h" not in payload:
        payload["h"] = payload.pop("horizon")

    # rolling_step -> backtest.step
    if "rolling_step" in payload:
        backtest = payload.get("backtest", {})
        if isinstance(backtest, BacktestSpec):
            backtest = backtest.model_dump()
        if isinstance(backtest, dict):
            backtest = dict(backtest)
            if "step" not in backtest:
                backtest["step"] = payload.pop("rolling_step")
            payload["backtest"] = backtest

    return payload


def _normalize_forecast_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy quantiles/levels to forecast_contract.

    Handles migration of standalone quantiles/levels fields to
    the nested forecast_contract structure.
    """
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
