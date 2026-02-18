"""TimesFM 2.5 TSFM adapter for tsagentkit.

Wraps Google's TimesFM 2.5 model for zero-shot time series forecasting.
Uses tsagentkit-timesfm library (pip install tsagentkit-timesfm) with google/timesfm-2.5-200m-pytorch model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.core.dataset import _normalize_freq_alias

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def load() -> Any:
    """Load pretrained TimesFM 2.5 200M model.

    Uses tsagentkit-timesfm library with google/timesfm-2.5-200m-pytorch model.
    Install: pip install tsagentkit-timesfm

    Returns:
        Loaded TimesFM model
    """
    import timesfm

    # Load TimesFM 2.5 200M model using from_pretrained
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    # Configure the model for inference
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        )
    )

    return model


def unload(model: Any | None = None) -> None:
    """Unload model resources (best-effort)."""
    del model


def fit(dataset: TSDataset) -> Any:
    """Fit TimesFM model (loads pretrained).

    TimesFM is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(
    model: Any, dataset: TSDataset, h: int, batch_size: int = 32
) -> pd.DataFrame:
    """Generate forecasts using TimesFM.

    Args:
        model: Loaded TimesFM model
        dataset: Time-series dataset
        h: Forecast horizon
        batch_size: Number of series to process in parallel

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np

    freq = _normalize_freq_alias(dataset.config.freq)

    # Group data by unique_id (dataset is pre-sorted by TSDataset)
    grouped = dataset.df.groupby("unique_id", sort=False)

    # Pre-extract all series data to avoid repeated groupby operations
    series_data = []
    for unique_id, group in grouped:
        context = group["y"].values.astype(np.float32)
        last_date = group["ds"].iloc[-1]
        series_data.append((unique_id, context, last_date))

    # TimesFM 2.5 requires sequences > 992 tokens to avoid NaN output
    # See: https://github.com/google-research/timesfm/issues/321
    min_context_length = 993

    forecasts = []
    # Process in batches
    for i in range(0, len(series_data), batch_size):
        batch = series_data[i : i + batch_size]

        # Prepare batched contexts with padding
        batch_contexts = []
        for _, context, _ in batch:
            if len(context) < min_context_length:
                padding_needed = min_context_length - len(context)
                context = np.pad(
                    context, (padding_needed, 0), mode="constant", constant_values=0
                )
            batch_contexts.append(context)

        # Batch inference: TimesFM accepts list of contexts
        point_forecasts, _ = model.forecast(horizon=h, inputs=batch_contexts)

        # Extract forecasts for each series in batch
        for j, (unique_id, _, last_date) in enumerate(batch):
            forecast_values = point_forecasts[j][:h]
            future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame({
                "unique_id": unique_id,
                "ds": future_dates[: len(forecast_values)],
                "yhat": forecast_values,
            })
            forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
