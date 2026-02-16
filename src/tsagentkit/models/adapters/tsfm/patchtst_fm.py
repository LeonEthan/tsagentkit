"""PatchTST-FM TSFM adapter for tsagentkit.

Wraps IBM's PatchTST-FM zero-shot model (`ibm-research/patchtst-fm-r1`).
Uses `tsfm_public.PatchTSTFMForPrediction` and keeps module-level caching.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.core.dataset import TSDataset

# Import torch for inference/device handling.
try:
    import torch
except ImportError:
    torch = None  # type: ignore


# Module-level cache for loaded model (ModelCache manages lifecycle).
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


def _resolve_device_map() -> str:
    """Select inference device for model loading.

    PatchTST-FM's internal autocast path can be unstable on MPS in some torch
    builds, so this adapter defaults to CPU when CUDA is unavailable.
    """
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_tsfm_public_import() -> None:
    """Ensure `tsfm_public` is importable from installed dependencies."""
    try:
        importlib.import_module("tsfm_public")
        return
    except ImportError:
        raise ImportError(
            "PatchTST-FM requires `tsfm_public`. Install `tsagentkit-patchtst-fm>=1.0.2`."
        ) from None


def _get_patchtst_fm_class() -> Any:
    """Return PatchTST-FM model class from tsfm_public."""
    _ensure_tsfm_public_import()
    module = importlib.import_module("tsfm_public")
    return module.PatchTSTFMForPrediction


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


def _resolve_context_length(model: Any) -> int | None:
    """Resolve expected context length from model config."""
    config = getattr(model, "config", None)
    context_length = getattr(config, "context_length", None)
    if isinstance(context_length, int) and context_length > 0:
        return context_length
    return None


def _sanitize_context(values: np.ndarray) -> np.ndarray:
    """Fill NaNs using mean-imputation (or zeros if all values are NaN)."""
    context = np.asarray(values, dtype=np.float32)
    if context.size == 0:
        return np.zeros(1, dtype=np.float32)

    if np.isnan(context).any():
        if np.isnan(context).all():
            context = np.zeros_like(context, dtype=np.float32)
        else:
            context = np.nan_to_num(context, nan=float(np.nanmean(context)))
    return context


def _left_pad_context(context: np.ndarray, context_length: int | None) -> np.ndarray:
    """Left-pad context to the model context length."""
    if context_length is None or len(context) >= context_length:
        return context

    fill_value = float(np.mean(context)) if context.size else 0.0
    pad = context_length - len(context)
    return np.pad(context, (pad, 0), mode="constant", constant_values=fill_value)


def _model_quantile_levels(model: Any) -> list[float] | None:
    """Return model quantile levels when available."""
    quantiles = getattr(getattr(model, "config", None), "quantile_levels", None)
    if quantiles is None:
        quantiles = getattr(getattr(model, "backbone", None), "quantile_levels", None)
    if quantiles is None:
        return None
    return [float(q) for q in quantiles]


def _inference_quantiles(model_quantiles: list[float] | None) -> list[float] | None:
    """Prefer single median quantile inference when supported."""
    if not model_quantiles:
        return None
    closest = min(model_quantiles, key=lambda q: abs(q - 0.5))
    if np.isclose(closest, 0.5):
        return [closest]
    return model_quantiles


def _median_index(quantile_levels: list[float] | None, size: int) -> int:
    """Pick median quantile index based on available levels."""
    if not quantile_levels:
        return size // 2
    return min(range(size), key=lambda i: abs(float(quantile_levels[i]) - 0.5))


def _to_numpy(values: Any) -> np.ndarray:
    """Convert tensor-like outputs into numpy arrays."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy(), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _extract_forecast_values(
    outputs: Any,
    h: int,
    quantile_levels: list[float] | None = None,
) -> np.ndarray:
    """Extract point forecasts from model outputs."""
    predictions = None

    if isinstance(outputs, tuple) and outputs:
        outputs = outputs[0]

    if hasattr(outputs, "quantile_predictions"):
        predictions = outputs.quantile_predictions
    elif hasattr(outputs, "prediction_outputs"):
        predictions = outputs.prediction_outputs
    elif isinstance(outputs, dict):
        if "quantile_predictions" in outputs:
            predictions = outputs["quantile_predictions"]
        else:
            predictions = outputs.get("prediction_outputs")
    else:
        predictions = outputs

    if predictions is None:
        raise ValueError("PatchTST output did not include forecast predictions.")

    arr = _to_numpy(predictions)
    quantile_count = len(quantile_levels) if quantile_levels is not None else None

    if arr.ndim == 1:
        values = arr[:h]
    elif arr.ndim == 2:
        if quantile_count is not None and arr.shape[0] == quantile_count:
            q_idx = _median_index(quantile_levels, arr.shape[0])
            values = arr[q_idx, :h]
        elif quantile_count is not None and arr.shape[1] == quantile_count:
            q_idx = _median_index(quantile_levels, arr.shape[1])
            values = arr[:h, q_idx]
        else:
            if arr.shape[0] == 1:
                values = arr[0, :h]
            elif arr.shape[1] == 1:
                values = arr[:h, 0]
            else:
                values = arr[0, :h]
    elif arr.ndim == 3:
        # Expected PatchTST-FM shape is (batch, quantile, horizon), but we keep
        # compatibility for horizon-first tensors.
        sample = arr[0]
        if quantile_count is not None and sample.shape[0] == quantile_count:
            q_idx = _median_index(quantile_levels, sample.shape[0])
            values = sample[q_idx, :h]
        elif quantile_count is not None and sample.shape[1] == quantile_count:
            q_idx = _median_index(quantile_levels, sample.shape[1])
            values = sample[:h, q_idx]
        else:
            # Fallback heuristic if quantile metadata is unavailable.
            if sample.shape[0] <= sample.shape[1]:
                q_idx = sample.shape[0] // 2
                values = sample[q_idx, :h]
            else:
                q_idx = sample.shape[1] // 2
                values = sample[:h, q_idx]
    else:
        raise ValueError(f"Unexpected PatchTST output rank: {arr.ndim}")

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(h, dtype=np.float32)
    if len(values) < h:
        return np.pad(values, (0, h - len(values)), mode="edge")
    return values[:h]


def load(model_name: str = "ibm-research/patchtst-fm-r1") -> Any:
    """Load pretrained PatchTST-FM model.

    Uses `tsfm_public.PatchTSTFMForPrediction`.

    Args:
        model_name: Model name (ibm-research/patchtst-fm-r1)

    Returns:
        Loaded PatchTST-FM model
    """
    global _loaded_model, _default_model_name

    if _loaded_model is None or _default_model_name != model_name:
        device_map = _resolve_device_map()
        model_cls = _get_patchtst_fm_class()

        try:
            _loaded_model = model_cls.from_pretrained(
                model_name,
                device_map=device_map,
            )
        except TypeError:
            _loaded_model = model_cls.from_pretrained(model_name)
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
    del dataset
    return load()


@contextmanager
def _mps_guard(model_device: Any | None):
    """Guard against unstable MPS checks inside PatchTST-FM internals."""
    if torch is None:
        yield
        return

    device_type = getattr(model_device, "type", str(model_device)) if model_device is not None else "cpu"
    if device_type == "mps":
        yield
        return

    torch_mps = getattr(torch, "mps", None)
    backend_mps = getattr(getattr(torch, "backends", None), "mps", None)

    if (
        torch_mps is not None
        and not hasattr(torch_mps, "is_available")
        and backend_mps is not None
        and hasattr(backend_mps, "is_available")
    ):
        torch_mps.is_available = backend_mps.is_available

    originals: list[tuple[Any, Any]] = []
    if torch_mps is not None and hasattr(torch_mps, "is_available"):
        originals.append((torch_mps, torch_mps.is_available))
        torch_mps.is_available = lambda: False
    if backend_mps is not None and hasattr(backend_mps, "is_available"):
        originals.append((backend_mps, backend_mps.is_available))
        backend_mps.is_available = lambda: False

    try:
        yield
    finally:
        for module_obj, original in originals:
            module_obj.is_available = original


def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate forecasts using PatchTST-FM.

    Args:
        model: Loaded PatchTST-FM model
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

    model_quantiles = _model_quantile_levels(model)
    requested_quantiles = _inference_quantiles(model_quantiles)
    model_device = _resolve_model_device(model)
    context_length = _resolve_context_length(model)

    # Process each series
    for unique_id in dataset.df[id_col].unique():
        mask = dataset.df[id_col] == unique_id
        series_df = dataset.df[mask].sort_values(time_col)
        context = series_df[target_col].to_numpy(dtype=np.float32)
        context = _sanitize_context(context)
        context = _left_pad_context(context, context_length)

        if model_device is None:
            context_tensor = torch.tensor(context, dtype=torch.float32)
        else:
            context_tensor = torch.tensor(context, dtype=torch.float32, device=model_device)

        # Generate forecast with PatchTST-FM API.
        with torch.no_grad(), _mps_guard(model_device):
            try:
                outputs = model(
                    inputs=[context_tensor],
                    prediction_length=h,
                    quantile_levels=requested_quantiles,
                    return_loss=False,
                )
            except TypeError:
                outputs = model(
                    inputs=[context_tensor],
                    prediction_length=h,
                    quantile_levels=requested_quantiles,
                )

        forecast_values = _extract_forecast_values(
            outputs=outputs,
            h=h,
            quantile_levels=requested_quantiles or model_quantiles,
        )
        if np.isnan(forecast_values).all():
            fallback = float(context[-1]) if context.size else 0.0
            forecast_values = np.full(h, fallback, dtype=np.float32)
        elif np.isnan(forecast_values).any():
            fill_value = np.nanmean(forecast_values)
            if np.isnan(fill_value):
                fill_value = float(context[-1]) if context.size else 0.0
            forecast_values = np.nan_to_num(forecast_values, nan=float(fill_value))

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
