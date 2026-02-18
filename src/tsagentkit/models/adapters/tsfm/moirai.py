"""Moirai 2.0 TSFM adapter for tsagentkit.

Wraps Salesforce's Moirai 2.0 model for zero-shot time series forecasting.
Uses tsagentkit-uni2ts library (pip install tsagentkit-uni2ts) with Salesforce/moirai-2.0-R-small model.
Moirai 2.0 uses a decoder-only architecture (different from Moirai 1.x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from tsagentkit.core.dataset import _normalize_freq_alias

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset


def load(model_name: str = "Salesforce/moirai-2.0-R-small", device: str | None = None) -> Any:
    """Load pretrained Moirai 2.0 model.

    Uses tsagentkit-uni2ts library with Salesforce/moirai-2.0-R-small model.
    Install: pip install tsagentkit-uni2ts

    Moirai 2.0 uses Moirai2Module and Moirai2Forecast (decoder-only architecture).

    Args:
        model_name: Moirai model variant (Salesforce/moirai-2.0-R-small, etc.)
        device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        Loaded Moirai model module
    """
    from uni2ts.model.moirai2 import Moirai2Module

    from tsagentkit.core.device import resolve_device

    resolved = resolve_device(device or "auto", allow_mps=True)

    # Load the pretrained module
    module = Moirai2Module.from_pretrained(model_name)

    # Move module to device if supported
    if hasattr(module, "to") and resolved in ("cuda", "mps", "cpu"):
        import torch

        module = module.to(torch.device(resolved))

    return {"model_name": model_name, "module": module}


def unload(model: Any | None = None) -> None:
    """Unload model resources (best-effort)."""
    del model


def fit(dataset: TSDataset) -> Any:
    """Fit Moirai model (loads pretrained).

    Moirai is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(
    model: Any,
    dataset: TSDataset,
    h: int,
    batch_size: int = 32,
    quantiles: tuple[float, ...] | list[float] | None = None,
) -> pd.DataFrame:
    """Generate forecasts using Moirai 2.0.

    Args:
        model: Loaded Moirai model (dict with model_name and module)
        dataset: Time-series dataset
        h: Forecast horizon
        batch_size: Number of series to process in parallel
        quantiles: Optional quantile levels to include as q{level} columns

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    import numpy as np
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai2 import Moirai2Forecast

    module = model["module"]
    freq = _normalize_freq_alias(dataset.config.freq)
    requested_quantiles = [float(q) for q in quantiles] if quantiles else []

    # Group data by unique_id (dataset is pre-sorted by TSDataset)
    grouped = dataset.df.groupby("unique_id", sort=False)

    def _sanitize_context(values: np.ndarray) -> np.ndarray:
        """Ensure target context is finite for GluonTS transforms."""
        if np.isfinite(values).all():
            return values

        cleaned = pd.Series(values, copy=True)
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.ffill().bfill().fillna(0.0)
        return cast(np.ndarray, cleaned.to_numpy(dtype=np.float32))

    # Check what covariates are available
    covariates_set = dataset.covariates
    has_future_covariates = False
    has_past_covariates = False

    if covariates_set is not None and not covariates_set.is_empty():
        has_future_covariates = covariates_set.future is not None
        has_past_covariates = covariates_set.past is not None

    # Get feature columns from covariates (excluding unique_id and ds)
    future_feature_cols: list[str] = []
    past_feature_cols: list[str] = []

    if has_future_covariates and covariates_set is not None and covariates_set.future is not None:
        cov_cols = covariates_set.future.columns.tolist()
        future_feature_cols = [c for c in cov_cols if c not in ("unique_id", "ds")]

    if has_past_covariates and covariates_set is not None and covariates_set.past is not None:
        cov_cols = covariates_set.past.columns.tolist()
        past_feature_cols = [c for c in cov_cols if c not in ("unique_id", "ds")]

    # Pre-extract all series data
    series_data = []
    for unique_id, group in grouped:
        context = _sanitize_context(group["y"].to_numpy(dtype=np.float32))
        last_date = group["ds"].iloc[-1]
        start_date = group["ds"].iloc[0]
        series_data.append((unique_id, context, last_date, start_date))

    forecasts = []
    # Process in batches
    for i in range(0, len(series_data), batch_size):
        batch = series_data[i : i + batch_size]

        # Determine max context length for this batch
        max_ctx_len = max(len(ctx) for _, ctx, _, _ in batch)

        # Calculate covariate dimensions
        feat_dynamic_real_dim = len(future_feature_cols)
        past_feat_dynamic_real_dim = len(past_feature_cols)

        # Create forecast model with batch-aware context length
        forecast_model = Moirai2Forecast(
            module=module,
            prediction_length=h,
            context_length=max_ctx_len,
            target_dim=1,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )

        # Use batch_size > 1 for efficient inference
        predictor = forecast_model.create_predictor(batch_size=len(batch))

        # Build multi-series PandasDataset with covariates
        ts_dict = {}
        for unique_id, context, _, start_date in batch:
            ts_index = pd.date_range(start=start_date, periods=len(context), freq=freq)

            # Check if we have covariates for this series
            has_series_future_cov = False
            if len(future_feature_cols) > 0:
                future_cov_df = dataset.get_covariates_for_series(unique_id, "future")
                has_series_future_cov = future_cov_df is not None and len(future_cov_df) >= h

            has_series_past_cov = False
            if len(past_feature_cols) > 0:
                past_cov_df = dataset.get_covariates_for_series(unique_id, "past")
                has_series_past_cov = past_cov_df is not None and len(past_cov_df) > 0

            # Build covariate DataFrame if we have any covariates
            if has_series_future_cov or has_series_past_cov:
                # Extend index to include forecast horizon for feat_dynamic_real
                if has_series_future_cov:
                    future_dates = pd.date_range(
                        start=ts_index[-1], periods=h + 1, freq=freq
                    )[1:]
                    full_index = ts_index.union(future_dates)
                else:
                    full_index = ts_index

                # Build covariate columns
                cov_data = {}

                # Add future covariates
                if has_series_future_cov and future_cov_df is not None:
                    for col_idx, col_name in enumerate(future_feature_cols):
                        cov_values = np.full(len(full_index), np.nan, dtype=np.float32)
                        future_values = future_cov_df[col_name].values[:h]
                        cov_values[len(context) : len(context) + len(future_values)] = future_values
                        cov_data[f"feat_dynamic_real_{col_idx}"] = cov_values

                # Add past covariates
                if has_series_past_cov and past_cov_df is not None:
                    for col_idx, col_name in enumerate(past_feature_cols):
                        cov_values = np.full(len(full_index), np.nan, dtype=np.float32)
                        series_past_cov = past_cov_df[past_cov_df["ds"].isin(ts_index)]
                        if len(series_past_cov) > 0:
                            aligned_values = series_past_cov.set_index("ds").reindex(ts_index)[
                                col_name
                            ].values
                            cov_values[:len(context)] = aligned_values
                        cov_data[f"past_feat_dynamic_real_{col_idx}"] = cov_values

                # Create target series and covariate DataFrame as separate entries
                ts_dict[unique_id] = pd.Series(context, index=ts_index)

                # Add covariates with extended index
                if cov_data:
                    cov_df = pd.DataFrame(cov_data, index=full_index)
                    # Merge covariates into ts_dict entry by converting Series to DataFrame
                    target_df = pd.DataFrame({"target": context}, index=ts_index)
                    if len(full_index) > len(ts_index):
                        # Need to extend target with NaN for forecast horizon
                        target_extended = np.full(len(full_index), np.nan, dtype=np.float32)
                        target_extended[:len(context)] = context
                        target_df = pd.DataFrame({"target": target_extended}, index=full_index)
                    # Merge covariates
                    for col in cov_df.columns:
                        target_df[col] = cov_df[col]
                    ts_dict[unique_id] = target_df
            else:
                # No covariates - just use Series
                ts_dict[unique_id] = pd.Series(context, index=ts_index)

        # PandasDataset with multiple series
        gts_dataset = PandasDataset(ts_dict)

        # Batch inference
        forecast_it = predictor.predict(gts_dataset)

        # Extract forecasts for each series
        for (unique_id, _, last_date, _), forecast in zip(batch, forecast_it, strict=False):
            forecast_values = forecast.quantile(0.5)
            future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame({
                "unique_id": unique_id,
                "ds": future_dates[: len(forecast_values)],
                "yhat": forecast_values,
            })
            for q in requested_quantiles:
                q_values = forecast.quantile(q)
                forecast_df[f"q{q}"] = q_values[: len(forecast_values)]
            forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)
