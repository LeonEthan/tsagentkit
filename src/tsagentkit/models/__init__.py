"""Models module for tsagentkit.

Minimal model fitting and prediction using statsforecast baselines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


def fit(dataset: TSDataset, model_name: str) -> Any:
    """Fit a statistical baseline model.

    Args:
        dataset: Time-series dataset
        model_name: Model name (e.g., 'SeasonalNaive', 'HistoricAverage', 'Naive')

    Returns:
        Fitted model artifact
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import (
            HistoricAverage,
            Naive,
            SeasonalNaive,
        )
    except ImportError:
        raise ImportError("statsforecast is required. Install with: pip install statsforecast")

    model_map = {
        "SeasonalNaive": SeasonalNaive,
        "HistoricAverage": HistoricAverage,
        "Naive": Naive,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")

    season_length = dataset.config.season_length or 1
    if model_name == "SeasonalNaive":
        model = model_map[model_name](season_length=season_length)
    else:
        model = model_map[model_name]()

    # Prepare data for statsforecast
    df = dataset.df.copy()
    df = df.rename(columns={
        dataset.config.id_col: "unique_id",
        dataset.config.time_col: "ds",
        dataset.config.target_col: "y",
    })

    # Fit model
    sf = StatsForecast(models=[model], freq=dataset.config.freq, n_jobs=1)
    sf.fit(df)

    return {"sf": sf, "model_name": model_name}


def predict(
    dataset: TSDataset,
    artifact: Any,
    h: int | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate forecasts from fitted model.

    Args:
        dataset: Time-series dataset
        artifact: Model artifact from fit()
        h: Forecast horizon (defaults to config.h)
        quantiles: Quantile levels for prediction intervals

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat, q0.1, ...]
    """
    sf = artifact["sf"]
    horizon = h or dataset.config.h

    # Generate point forecast
    forecast = sf.predict(h=horizon)

    # Rename columns back
    forecast = forecast.rename(columns={
        artifact["model_name"]: "yhat",
    })

    # Add quantile forecasts if requested
    if quantiles:
        from statsforecast.models import Naive
        # Get historical data for std calculation
        df = dataset.df.copy()
        df = df.rename(columns={
            dataset.config.id_col: "unique_id",
            dataset.config.time_col: "ds",
            dataset.config.target_col: "y",
        })
        # Calculate historical std per series for uncertainty
        hist_std = df.groupby("unique_id")["y"].std().fillna(0)
        yhat = forecast["yhat"]
        unique_ids = forecast["unique_id"]

        for q in quantiles:
            q_col = f"q{q}"
            # Approximate quantile: mean + z_score * std
            # z_score for quantile q: norm.ppf(q)
            try:
                from scipy.stats import norm
                z = norm.ppf(q)
            except ImportError:
                # Fallback to rough approximation
                z = {0.1: -1.28, 0.2: -0.84, 0.3: -0.52, 0.4: -0.25, 0.5: 0,
                     0.6: 0.25, 0.7: 0.52, 0.8: 0.84, 0.9: 1.28}.get(q, 0)
            # Map std to each row based on unique_id
            std_map = hist_std.reindex(unique_ids).values
            forecast[q_col] = yhat + z * std_map

    return forecast.reset_index(drop=True)


def fit_tsfm(dataset: TSDataset, adapter_name: str) -> Any:
    """Fit a TSFM model via adapter.

    Args:
        dataset: Time-series dataset
        adapter_name: Adapter name (e.g., 'chronos', 'moirai', 'timesfm')

    Returns:
        Fitted model artifact
    """
    adapter_map = {
        "chronos": "tsagentkit.models.adapters.chronos",
        "timesfm": "tsagentkit.models.adapters.timesfm",
        "moirai": "tsagentkit.models.adapters.moirai",
    }

    if adapter_name not in adapter_map:
        raise ValueError(f"Unknown TSFM adapter: {adapter_name}. Available: {list(adapter_map.keys())}")

    try:
        module = __import__(adapter_map[adapter_name], fromlist=["fit"])
        return module.fit(dataset)
    except ImportError as e:
        raise ImportError(
            f"TSFM adapter '{adapter_name}' not available. "
            f"Install the appropriate package: {e}"
        )


def predict_tsfm(
    dataset: TSDataset,
    artifact: Any,
    h: int | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate forecasts from TSFM model.

    Args:
        dataset: Time-series dataset
        artifact: Model artifact from fit_tsfm()
        h: Forecast horizon (defaults to config.h)
        quantiles: Quantile levels for prediction intervals

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat, q0.1, ...]
    """
    # The artifact contains the adapter reference
    if "adapter" in artifact:
        predict_fn = getattr(artifact["adapter"], "predict", None)
        if predict_fn is None:
            raise ValueError("Invalid TSFM adapter - missing predict method")
        # Check if predict supports quantiles
        import inspect
        sig = inspect.signature(predict_fn)
        if "quantiles" in sig.parameters:
            return artifact["adapter"].predict(dataset, artifact, h, quantiles=quantiles)
        else:
            return artifact["adapter"].predict(dataset, artifact, h)

    raise ValueError("Invalid TSFM artifact - missing adapter reference")


__all__ = ["fit", "predict", "fit_tsfm", "predict_tsfm"]
