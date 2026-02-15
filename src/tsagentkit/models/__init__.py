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
) -> pd.DataFrame:
    """Generate forecasts from fitted model.

    Args:
        dataset: Time-series dataset
        artifact: Model artifact from fit()
        h: Forecast horizon (defaults to config.h)

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    sf = artifact["sf"]
    horizon = h or dataset.config.h

    # Generate forecast
    forecast = sf.predict(h=horizon)

    # Rename columns back
    forecast = forecast.rename(columns={
        artifact["model_name"]: "yhat",
    })

    return forecast.reset_index()


def fit_tsfm(dataset: TSDataset, adapter_name: str) -> Any:
    """Fit a TSFM model via adapter.

    Args:
        dataset: Time-series dataset
        adapter_name: Adapter name (e.g., 'chronos', 'moirai', 'timesfm')

    Returns:
        Fitted model artifact
    """
    raise NotImplementedError(
        f"TSFM adapter '{adapter_name}' not available. "
        "Install the appropriate package (chronos, moirai, timesfm)."
    )


def predict_tsfm(
    dataset: TSDataset,
    artifact: Any,
    h: int | None = None,
) -> pd.DataFrame:
    """Generate forecasts from TSFM model.

    Args:
        dataset: Time-series dataset
        artifact: Model artifact from fit_tsfm()
        h: Forecast horizon (defaults to config.h)

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    raise NotImplementedError("TSFM prediction not available without adapters.")


__all__ = ["fit", "predict", "fit_tsfm", "predict_tsfm"]
