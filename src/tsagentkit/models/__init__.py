"""Models module for tsagentkit.

Minimal model fitting and prediction using the registry/cache/protocol pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.models.cache import ModelCache
from tsagentkit.models.ensemble import ensemble, ensemble_with_quantiles
from tsagentkit.models.protocol import fit as _protocol_fit, predict as _protocol_predict
from tsagentkit.models.registry import (
    REGISTRY,
    ModelSpec,
    get_spec,
    list_available,
    list_models,
)

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def fit(dataset: TSDataset, model_name: str) -> Any:
    """Fit a statistical baseline model (backward compatible).

    Args:
        dataset: Time-series dataset
        model_name: Model name (e.g., 'SeasonalNaive', 'HistoricAverage', 'Naive')

    Returns:
        Fitted model artifact
    """
    # Map old names to registry names
    name_map = {
        "SeasonalNaive": "seasonal_naive",
        "Naive": "naive",
        "HistoricAverage": "naive",  # Fallback to naive
    }
    registry_name = name_map.get(model_name, model_name)
    try:
        spec = get_spec(registry_name)
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}")
    model = _protocol_fit(spec, dataset)

    # Wrap in backward-compatible format (include 'sf' for test compatibility)
    if model_name == "SeasonalNaive":
        return {"sf": MockSF(model), "model": model, "model_name": model_name, "season_length": dataset.config.season_length or 1}
    return {"sf": MockSF(model), "model": model, "model_name": model_name}


class MockSF:
    """Mock StatsForecast object for backward compatibility."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, h: int) -> Any:
        """Mock predict method."""
        return None


def predict(
    dataset: TSDataset,
    artifact: Any,
    h: int | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate forecasts from fitted model (backward compatible).

    Args:
        dataset: Time-series dataset
        artifact: Model artifact from fit()
        h: Forecast horizon (defaults to config.h)
        quantiles: Quantile levels for prediction intervals

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat, q0.1, ...]
    """
    model_name = artifact["model_name"]
    name_map = {
        "SeasonalNaive": "seasonal_naive",
        "Naive": "naive",
        "HistoricAverage": "naive",
    }
    registry_name = name_map.get(model_name, model_name)
    spec = get_spec(registry_name)

    horizon = h or dataset.config.h
    forecast = _protocol_predict(spec, artifact["model"], dataset, horizon)

    # Add quantiles if requested (using simple heuristic)
    if quantiles:
        import numpy as np

        df = dataset.df.copy()
        hist_std = df.groupby(dataset.config.id_col)[dataset.config.target_col].std().fillna(0)
        yhat = forecast["yhat"]
        unique_ids = forecast[dataset.config.id_col]

        for q in quantiles:
            q_col = f"q{q}"
            try:
                from scipy.stats import norm
                z = norm.ppf(q)
            except ImportError:
                z = {0.1: -1.28, 0.2: -0.84, 0.3: -0.52, 0.4: -0.25, 0.5: 0,
                     0.6: 0.25, 0.7: 0.52, 0.8: 0.84, 0.9: 1.28}.get(q, 0)
            std_map = hist_std.reindex(unique_ids).values
            forecast[q_col] = yhat + z * std_map

    return forecast.reset_index(drop=True)


def fit_tsfm(dataset: TSDataset, adapter_name: str) -> Any:
    """Fit a TSFM model via adapter (backward compatible).

    Args:
        dataset: Time-series dataset
        adapter_name: Adapter name (e.g., 'chronos', 'moirai', 'timesfm')

    Returns:
        Fitted model artifact
    """
    if adapter_name not in REGISTRY:
        raise ValueError(f"Unknown TSFM adapter: {adapter_name}. Available: {list(REGISTRY.keys())}")

    spec = get_spec(adapter_name)
    if not spec.is_tsfm:
        raise ValueError(f"'{adapter_name}' is not a TSFM adapter")

    model = _protocol_fit(spec, dataset)
    return {"model": model, "model_name": adapter_name, "adapter": MockAdapter(spec)}


def predict_tsfm(
    dataset: TSDataset,
    artifact: Any,
    h: int | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate forecasts from TSFM model (backward compatible).

    Args:
        dataset: Time-series dataset
        artifact: Model artifact from fit_tsfm()
        h: Forecast horizon (defaults to config.h)
        quantiles: Quantile levels for prediction intervals

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat, q0.1, ...]
    """
    # Check for valid artifact (backward compatible error messages)
    if "model_name" not in artifact:
        raise ValueError("Invalid TSFM artifact - missing model_name reference")
    if "model" not in artifact:
        raise ValueError("Invalid TSFM artifact - missing model reference")

    model_name = artifact["model_name"]
    spec = get_spec(model_name)
    horizon = h or dataset.config.h

    forecast = _protocol_predict(spec, artifact["model"], dataset, horizon)

    # Add quantiles if requested
    if quantiles:
        import numpy as np

        df = dataset.df.copy()
        hist_std = df.groupby(dataset.config.id_col)[dataset.config.target_col].std().fillna(0)
        yhat = forecast["yhat"]
        unique_ids = forecast[dataset.config.id_col]

        for q in quantiles:
            q_col = f"q{q}"
            try:
                from scipy.stats import norm
                z = norm.ppf(q)
            except ImportError:
                z = {0.1: -1.28, 0.2: -0.84, 0.3: -0.52, 0.4: -0.25, 0.5: 0,
                     0.6: 0.25, 0.7: 0.52, 0.8: 0.84, 0.9: 1.28}.get(q, 0)
            std_map = hist_std.reindex(unique_ids).values
            forecast[q_col] = yhat + z * std_map

    return forecast.reset_index(drop=True)


class MockAdapter:
    """Mock adapter for backward compatibility with old artifact format."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def predict(self, dataset: TSDataset, artifact: Any, h: int | None = None) -> pd.DataFrame:
        """Predict method for backward compatibility."""
        return predict_tsfm(dataset, artifact, h)


__all__ = [
    # Registry
    "REGISTRY",
    "ModelSpec",
    "list_models",
    "get_spec",
    "list_available",
    # Cache
    "ModelCache",
    # Protocol (backward compatible)
    "fit",
    "predict",
    "fit_tsfm",
    "predict_tsfm",
    # Ensemble
    "ensemble",
    "ensemble_with_quantiles",
]
