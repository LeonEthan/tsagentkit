"""Models module for tsagentkit.

Provides model fitting, prediction, and TSFM (Time-Series Foundation Model)
adapters for various forecasting backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.contracts import ModelArtifact

# Import adapters submodules
from tsagentkit.models import adapters

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset


def fit(model_name: str, dataset: TSDataset, config: dict[str, Any]) -> ModelArtifact:
    """Fit a model.

    This is a stub implementation for v0.1.
    In a full implementation, this would dispatch to model-specific fit functions.

    Args:
        model_name: Name of the model to fit
        dataset: TSDataset with training data
        config: Model configuration

    Returns:
        ModelArtifact with fitted model
    """
    return ModelArtifact(
        model={"name": model_name, "config": config},
        model_name=model_name,
        config=config,
    )


def predict(
    model: ModelArtifact,
    dataset: TSDataset,
    horizon: int,
) -> pd.DataFrame:
    """Generate predictions.

    This is a stub implementation for v0.1.
    In a full implementation, this would dispatch to model-specific predict functions.

    Args:
        model: Fitted model artifact
        dataset: TSDataset with historical data
        horizon: Forecast horizon

    Returns:
        DataFrame with predictions
    """
    # Generate simple naive forecast for stub
    unique_ids = dataset.df["unique_id"].unique()
    last_dates = dataset.df.groupby("unique_id")["ds"].max()

    predictions = []
    for uid in unique_ids:
        last_date = last_dates[uid]
        last_value = dataset.df[
            (dataset.df["unique_id"] == uid) & (dataset.df["ds"] == last_date)
        ]["y"].values[0]

        for h in range(1, horizon + 1):
            predictions.append(
                {
                    "unique_id": uid,
                    "ds": pd.date_range(start=last_date, periods=h + 1, freq="D")[-1],
                    "yhat": last_value,
                }
            )

    return pd.DataFrame(predictions)


__all__ = ["fit", "predict", "adapters"]
