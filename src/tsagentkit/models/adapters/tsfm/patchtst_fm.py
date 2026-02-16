"""PatchTST TSFM adapter for tsagentkit.

Wraps IBM's PatchTST Foundation Model for zero-shot time series forecasting.
Uses Transformers PatchTSTForPrediction with ibm-research/patchtst-fm-r1 model.
Uses module-level caching for efficient model reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset

# Import torch for type hints and device detection
try:
    import torch
except ImportError:
    torch = None  # type: ignore


# Module-level cache for the loaded model
# ModelCache manages the lifecycle, this is just the storage
_loaded_model: Any | None = None
_default_model_name: str = "ibm-research/patchtst-fm-r1"


def _is_mps_available() -> bool:
    """Check MPS availability across torch variants."""
    if torch is None:
        return False

    torch_mps = getattr(torch, "mps", None)
    if torch_mps is not None and hasattr(torch_mps, "is_available"):
        return bool(torch_mps.is_available())

    backends = getattr(torch, "backends", None)
    if backends is not None:
        backend_mps = getattr(backends, "mps", None)
        if backend_mps is not None and hasattr(backend_mps, "is_available"):
            return bool(backend_mps.is_available())

    return False


def _resolve_model_device(model: Any) -> Any | None:
    """Best-effort lookup of model device for tensor placement."""
    if torch is None:
        return None

    try:
        if hasattr(model, "device"):
            return model.device

        if hasattr(model, "parameters"):
            return next(model.parameters()).device
    except Exception:
        return None

    return None


def _to_numpy(values: Any) -> np.ndarray:
    """Convert tensor-like outputs into numpy arrays."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy(), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _extract_forecast_values(outputs: Any, h: int, quantile_count: int | None = None) -> np.ndarray:
    """Extract median forecast from PatchTST outputs with schema compatibility."""
    predictions = None
    is_quantile_output = False

    if isinstance(outputs, tuple) and outputs:
        outputs = outputs[0]

    if hasattr(outputs, "quantile_predictions"):
        predictions = outputs.quantile_predictions
        is_quantile_output = True
    elif hasattr(outputs, "prediction_outputs"):
        predictions = outputs.prediction_outputs
    elif isinstance(outputs, dict):
        if "quantile_predictions" in outputs:
            predictions = outputs["quantile_predictions"]
            is_quantile_output = True
        elif "quantile_outputs" in outputs:
            predictions = outputs["quantile_outputs"]
            is_quantile_output = True
        else:
            predictions = outputs.get("prediction_outputs")
    else:
        predictions = outputs

    if predictions is None:
        raise ValueError("PatchTST output did not include forecast predictions.")

    arr = _to_numpy(predictions)
    is_quantile_like = is_quantile_output or (
        quantile_count is not None and quantile_count in arr.shape and quantile_count > 1
    )

    if arr.ndim == 1:
        values = arr[:h]
    elif arr.ndim == 2:
        if is_quantile_like:
            if quantile_count is not None and arr.shape[0] == quantile_count:
                values = arr[arr.shape[0] // 2, :h]
            elif quantile_count is not None and arr.shape[1] == quantile_count:
                values = arr[:h, arr.shape[1] // 2]
            else:
                values = arr[0, :h]
        else:
            if arr.shape[0] == 1:
                values = arr[0, :h]
            elif arr.shape[1] == 1:
                values = arr[:h, 0]
            else:
                values = arr[0, :h]
    elif arr.ndim == 3:
        sample = arr[0]
        if is_quantile_like:
            quantile_axis = 0
            if quantile_count is not None:
                if sample.shape[0] == quantile_count:
                    quantile_axis = 0
                elif sample.shape[1] == quantile_count:
                    quantile_axis = 1

            if quantile_axis == 0:
                median_idx = sample.shape[0] // 2
                values = sample[median_idx, :h]
            else:
                median_idx = sample.shape[1] // 2
                values = sample[:h, median_idx]
        else:
            # Point forecasts are typically (batch, horizon, channels).
            if sample.shape[1] == 1:
                values = sample[:h, 0]
            elif sample.shape[0] == 1:
                values = sample[0, :h]
            else:
                values = sample[:h, 0]
    else:
        raise ValueError(f"Unexpected PatchTST output rank: {arr.ndim}")

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(h, dtype=np.float32)
    if len(values) < h:
        return np.pad(values, (0, h - len(values)), mode="edge")
    return values[:h]


def load(model_name: str = "ibm-research/patchtst-fm-r1") -> Any:
    """Load pretrained PatchTST model.

    Uses Transformers PatchTSTForPrediction from transformers package.
    Install: pip install transformers torch

    Args:
        model_name: Model name (ibm-research/patchtst-fm-r1)

    Returns:
        Loaded PatchTSTForPrediction model
    """
    global _loaded_model, _default_model_name

    if _loaded_model is None or _default_model_name != model_name:
        # Determine device using module-level torch
        if torch is not None:
            if torch.cuda.is_available():
                device_map = "cuda"
            elif _is_mps_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        else:
            device_map = "cpu"

        from transformers import PatchTSTForPrediction

        try:
            _loaded_model = PatchTSTForPrediction.from_pretrained(
                model_name,
                device_map=device_map,
            )
        except TypeError:
            _loaded_model = PatchTSTForPrediction.from_pretrained(model_name)
            if hasattr(_loaded_model, "to"):
                _loaded_model = _loaded_model.to(device_map)

        if hasattr(_loaded_model, "eval"):
            _loaded_model.eval()
        _default_model_name = model_name

    return _loaded_model


def unload() -> None:
    """Unload model to free memory."""
    global _loaded_model
    _loaded_model = None


def fit(dataset: TSDataset) -> Any:
    """Fit PatchTST-FM model (loads pretrained).

    PatchTST-FM is a zero-shot model, so fit() just loads the model.

    Args:
        dataset: Time-series dataset (used for validation only)

    Returns:
        Loaded model
    """
    return load()


def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate forecasts using PatchTST-FM.

    Args:
        model: Loaded PatchTSTForPrediction model
        dataset: Time-series dataset
        h: Forecast horizon

    Returns:
        Forecast DataFrame with columns [unique_id, ds, yhat]
    """
    if torch is None:
        raise ImportError("PyTorch is required for PatchTST-FM inference.")

    forecasts = []
    id_col = dataset.config.id_col
    time_col = dataset.config.time_col
    target_col = dataset.config.target_col

    quantile_levels = getattr(getattr(model, "config", None), "quantile_levels", None)
    if quantile_levels is None:
        quantile_levels = getattr(getattr(model, "config", None), "quantiles", None)
    if quantile_levels is None:
        quantile_levels = getattr(getattr(model, "backbone", None), "quantile_levels", None)
    quantile_count = len(quantile_levels) if quantile_levels is not None else None
    model_device = _resolve_model_device(model)

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].to_numpy(dtype=np.float32)

        if model_device is None:
            context_tensor = torch.tensor(context, dtype=torch.float32)
        else:
            context_tensor = torch.tensor(context, dtype=torch.float32, device=model_device)

        # Generate forecast
        with torch.no_grad():
            past_values = context_tensor.unsqueeze(0).unsqueeze(-1)
            outputs = model(past_values=past_values)

        forecast_values = _extract_forecast_values(
            outputs=outputs,
            h=h,
            quantile_count=quantile_count,
        )

        # Generate future timestamps
        last_date = series_df[time_col].iloc[-1]
        freq = dataset.config.freq
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            id_col: unique_id,
            time_col: future_dates[:len(forecast_values)],
            "yhat": forecast_values,
        })
        forecasts.append(forecast_df)

    return pd.concat(forecasts, ignore_index=True)


# Backward compatibility wrapper
class PatchTSTFMAdapter:
    """Backward-compatible wrapper for PatchTST-FM adapter.

    DEPRECATED: Use module-level functions directly:
        from tsagentkit.models.adapters.tsfm.patchtst_fm import load, fit, predict
        model = load(model_name="ibm-research/patchtst-fm-r1")
        artifact = fit(dataset)  # or just use load()
        forecast = predict(artifact, dataset, h=7)
    """

    def __init__(self, model_name: str = "ibm-research/patchtst-fm-r1"):
        self.model_name = model_name

    def fit(self, dataset: TSDataset) -> dict[str, Any]:
        """Load model and return artifact."""
        model = load(self.model_name)
        return {"model": model, "model_name": self.model_name, "adapter": self}

    def predict(self, dataset: TSDataset, artifact: dict[str, Any], h: int) -> pd.DataFrame:
        """Generate forecasts."""
        model = artifact.get("model", _loaded_model)
        if model is None:
            model = load(self.model_name)
        return predict(model, dataset, h)
