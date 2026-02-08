"""API discovery and introspection for tsagentkit.

Provides ``describe()`` which returns a machine-readable schema of
the library's public surface: version, stable APIs, error codes with
fix hints, installation tiers, and TSFM adapter status.

Usage:
    >>> from tsagentkit import describe
    >>> info = describe()
    >>> info["version"]
    '1.1.2'
"""

from __future__ import annotations

from typing import Any


def describe() -> dict[str, Any]:
    """Return a machine-readable API schema for tsagentkit.

    Returns a dictionary with:
      - ``version``: library version string
      - ``apis``: mapping of task names to primary API functions
      - ``error_codes``: mapping of error codes to message/fix_hint
      - ``install_tiers``: mapping of extra names to descriptions
      - ``tsfm_adapters``: list of adapter dicts with availability

    Returns:
        Structured dict describing the full public surface.
    """
    import tsagentkit

    return {
        "version": tsagentkit.__version__,
        "apis": _get_apis(),
        "error_codes": _get_error_codes(),
        "install_tiers": _get_install_tiers(),
        "tsfm_adapters": _get_tsfm_adapters(),
    }


def _get_apis() -> dict[str, dict[str, str]]:
    """Return stable assembly API surface."""
    return {
        "validate": {
            "function": "validate_contract",
            "description": "Validate input DataFrame against panel contract",
        },
        "qa": {
            "function": "run_qa",
            "description": "Run data quality checks and optional repairs",
        },
        "align_covariates": {
            "function": "align_covariates",
            "description": "Align covariates with target series (leakage-safe)",
        },
        "build_dataset": {
            "function": "build_dataset",
            "description": "Construct immutable TSDataset from DataFrame",
        },
        "make_plan": {
            "function": "make_plan",
            "description": "Generate deterministic routing plan with candidate models",
        },
        "backtest": {
            "function": "rolling_backtest",
            "description": "Rolling temporal cross-validation",
        },
        "fit": {
            "function": "fit",
            "description": "Fit model(s) according to plan",
        },
        "predict": {
            "function": "predict",
            "description": "Generate forecasts from fitted model",
        },
        "calibrate": {
            "function": "fit_calibrator / apply_calibrator",
            "description": "Conformal calibration of prediction intervals",
        },
        "package": {
            "function": "package_run",
            "description": "Package forecast results into RunArtifact with provenance",
        },
        "save_artifact": {
            "function": "save_run_artifact",
            "description": "Persist RunArtifact to JSON",
        },
        "load_artifact": {
            "function": "load_run_artifact",
            "description": "Load RunArtifact from JSON with schema checks",
        },
        "validate_for_serving": {
            "function": "validate_run_artifact_for_serving",
            "description": "Validate artifact signatures for serving gate",
        },
        "replay": {
            "function": "replay_forecast_from_artifact",
            "description": "Deterministic replay from artifact",
        },
        "forecast": {
            "function": "forecast",
            "description": "Zero-config convenience wrapper (quickstart)",
        },
        "diagnose": {
            "function": "diagnose",
            "description": "Dry-run validation + QA + routing without fitting",
        },
        "repair": {
            "function": "repair",
            "description": "Auto-fix common data issues based on error code",
        },
        "run_forecast": {
            "function": "run_forecast",
            "description": "Full pipeline convenience wrapper",
        },
    }


def _get_error_codes() -> dict[str, dict[str, str]]:
    """Return all error codes with descriptions and fix hints."""
    from tsagentkit.contracts.errors import ERROR_REGISTRY

    result: dict[str, dict[str, str]] = {}
    for code, cls in ERROR_REGISTRY.items():
        result[code] = {
            "class": cls.__name__,
            "description": cls.__doc__ or "",
            "fix_hint": cls.fix_hint if hasattr(cls, "fix_hint") else "",
        }
    return result


def _get_install_tiers() -> dict[str, dict[str, str]]:
    """Return installation tier descriptions."""
    return {
        "core": {
            "install": "pip install tsagentkit",
            "description": "Statistical baselines (statsforecast) + full pipeline. No GPU required.",
        },
        "tsfm": {
            "install": "pip install tsagentkit[tsfm]",
            "description": "Chronos, Moirai, TimesFM foundation model adapters. Requires PyTorch.",
        },
        "hierarchy": {
            "install": "pip install tsagentkit[hierarchy]",
            "description": "Hierarchical forecast reconciliation (hierarchicalforecast).",
        },
        "features": {
            "install": "pip install tsagentkit[features]",
            "description": "Feature engineering with tsfeatures, tsfresh, and sktime.",
        },
        "full": {
            "install": "pip install tsagentkit[full]",
            "description": "All optional dependencies (tsfm + hierarchy + features).",
        },
    }


def _get_tsfm_adapters() -> list[dict[str, Any]]:
    """Return TSFM adapter registration and availability status."""
    adapters: list[dict[str, Any]] = []

    try:
        from tsagentkit.models.adapters import AdapterRegistry

        for name in AdapterRegistry.list_available():
            is_available, reason = AdapterRegistry.check_availability(name)
            adapters.append(
                {
                    "name": name,
                    "registered": True,
                    "available": is_available,
                    "reason": reason or None,
                }
            )
    except Exception:
        # If models can't be imported, report empty
        pass

    return adapters


__all__ = ["describe"]
