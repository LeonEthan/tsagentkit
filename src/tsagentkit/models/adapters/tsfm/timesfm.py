"""TimesFM 2.5 TSFM adapter for tsagentkit.

Wraps Google's TimesFM 2.5 model for zero-shot time series forecasting.
Uses tsagentkit-timesfm library (pip install tsagentkit-timesfm) with google/timesfm-2.5-200m-pytorch model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tsagentkit.core.dataset import _normalize_freq_alias
from tsagentkit.models.length_utils import adjust_context_length, validate_prediction_length

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset

logger = logging.getLogger(__name__)


def load(
    device: str | None = None,
    spec: Any | None = None,
) -> Any:
    """Load pretrained TimesFM 2.5 200M model.

    Uses tsagentkit-timesfm library with google/timesfm-2.5-200m-pytorch model.
    Install: pip install tsagentkit-timesfm

    Args:
        device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)
        spec: Optional ModelSpec with length limits (uses registry if not provided)

    Returns:
        Loaded TimesFM model
    """
    import timesfm

    from tsagentkit.core.device import resolve_device
    from tsagentkit.models.registry import get_spec

    resolved = resolve_device(device or "auto", allow_mps=True)

    # Get spec for limits (if not provided)
    if spec is None:
        spec = get_spec("timesfm")

    # Load TimesFM 2.5 200M model using from_pretrained
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    # Move model to device if supported
    if hasattr(model, "to") and resolved in ("cuda", "mps", "cpu"):
        import torch

        model = model.to(torch.device(resolved))

    # Use registry limits with fallback to defaults
    max_context = spec.max_context_length or 16384
    max_horizon = spec.max_prediction_length or 1024

    # Configure the model for inference
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        )
    )

    return model


def unload(model: Any | None = None) -> None:
    """Unload model resources (best-effort)."""
    del model


def fit(
    dataset: TSDataset,
    spec: Any | None = None,
) -> Any:
    """Fit TimesFM model (loads pretrained).

    TimesFM is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)
        spec: Optional ModelSpec with length limits

    Returns:
        Loaded model
    """
    return load(spec=spec)


def _format_qcol(q: float) -> str:
    """Format quantile column name."""
    return f"q{q}"


def _model_quantiles(model: Any) -> list[float] | None:
    """Return TimesFM quantile levels when available."""
    quantiles = getattr(getattr(getattr(model, "model", None), "config", None), "quantiles", None)
    if quantiles is None:
        quantiles = getattr(model, "quantiles", None)
    if quantiles is None:
        return None
    return [float(q) for q in quantiles]


def _quantile_index(
    q: float,
    qdim: int,
    model_quantiles: list[float] | None,
) -> int | None:
    """Resolve quantile index in TimesFM quantile forecast tensor."""
    import numpy as np

    if qdim <= 0:
        return None

    # Preferred mapping: known model quantile levels.
    if model_quantiles:
        matches = [i for i, mq in enumerate(model_quantiles) if np.isclose(mq, q)]
        if matches:
            i = matches[0]
            if qdim == len(model_quantiles):
                return i
            if qdim == len(model_quantiles) + 1:
                return i + 1

    # Fallback mapping for common TimesFM outputs:
    # qdim=10 usually corresponds to deciles with 0.5 at index 5.
    if qdim == 10:
        return int(np.clip(round(q * 10), 0, 9))
    if qdim == 9:
        return int(np.clip(round(q * 10) - 1, 0, 8))
    return None


def _extract_requested_quantiles(
    quantile_forecasts: Any,
    series_idx: int,
    h: int,
    requested_quantiles: list[float],
    model_quantiles: list[float] | None,
) -> dict[str, Any]:
    """Extract q{level} arrays for one series from TimesFM quantile output."""
    import numpy as np

    if not requested_quantiles or quantile_forecasts is None:
        return {}

    arr = np.asarray(quantile_forecasts)
    if arr.ndim != 3 or series_idx >= arr.shape[0]:
        return {}

    # Expected shape is (batch, horizon, qdim)
    sample = arr[series_idx]
    if sample.ndim != 2:
        return {}

    if sample.shape[0] >= h:
        horizon_first = True
        qdim = sample.shape[1]
    else:
        horizon_first = False
        qdim = sample.shape[0]

    extracted: dict[str, Any] = {}
    for q in requested_quantiles:
        q_idx = _quantile_index(q=q, qdim=qdim, model_quantiles=model_quantiles)
        if q_idx is None:
            continue

        if horizon_first:
            if q_idx >= sample.shape[1]:
                continue
            values = sample[:h, q_idx]
        else:
            if q_idx >= sample.shape[0]:
                continue
            values = sample[q_idx, :h]
        extracted[_format_qcol(q)] = values

    return extracted


def predict(
    model: Any,
    dataset: TSDataset,
    h: int,
    batch_size: int = 32,
    quantiles: tuple[float, ...] | list[float] | None = None,
    spec: Any | None = None,
) -> pd.DataFrame:
    """Generate forecasts using TimesFM.

    Args:
        model: Loaded TimesFM model
        dataset: Time-series dataset
        h: Forecast horizon
        batch_size: Number of series to process in parallel
        quantiles: Optional quantile levels to include as q{level} columns
        spec: Optional ModelSpec with length limits (uses registry if not provided)

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    from tsagentkit.models.registry import get_spec

    if spec is None:
        spec = get_spec("timesfm")

    freq = _normalize_freq_alias(dataset.config.freq)
    requested_quantiles = [float(q) for q in quantiles] if quantiles else []
    available_model_quantiles = _model_quantiles(model)

    # Validate horizon against model limits
    h = validate_prediction_length(h, spec, strict=False)

    # Group data by unique_id (dataset is pre-sorted by TSDataset)
    grouped = dataset.df.groupby("unique_id", sort=False)

    # Pre-extract all series data to avoid repeated groupby operations
    series_data = []
    for unique_id, group in grouped:
        context = group["y"].values.astype(np.float32)
        last_date = group["ds"].iloc[-1]
        series_data.append((unique_id, context, last_date))

    forecasts = []
    # Process in batches
    for i in range(0, len(series_data), batch_size):
        batch = series_data[i : i + batch_size]

        # Prepare batched contexts with centralized length adjustment
        batch_contexts = []
        for _, context, _ in batch:
            adjusted = adjust_context_length(
                context,
                spec,
                pad_value=0.0,  # TimesFM uses 0-padding
            )
            batch_contexts.append(adjusted.data)

            if adjusted.was_padded or adjusted.was_truncated:
                logger.debug(
                    f"Adjusted context: "
                    f"{adjusted.original_length} -> {adjusted.adjusted_length} "
                    f"(padded: {adjusted.padding_amount}, truncated: {adjusted.truncation_amount})"
                )

        # Batch inference: TimesFM accepts list of contexts
        point_forecasts, quantile_forecasts = model.forecast(horizon=h, inputs=batch_contexts)

        # Extract forecasts for each series in batch
        for j, (unique_id, _, last_date) in enumerate(batch):
            forecast_values = point_forecasts[j][:h]
            future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_dates[: len(forecast_values)],
                    "yhat": forecast_values,
                }
            )

            extracted = _extract_requested_quantiles(
                quantile_forecasts=quantile_forecasts,
                series_idx=j,
                h=h,
                requested_quantiles=requested_quantiles,
                model_quantiles=available_model_quantiles,
            )
            for col, values in extracted.items():
                forecast_df[col] = values[: len(forecast_values)]

            forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
