"""Generic output parsing utilities for TSFM adapters.

Pure functions for tensor conversion, quantile resolution, and forecast extraction.
Used by PatchTST-FM, TimesFM, and other TSFM adapters to eliminate duplicate logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def tensor_to_numpy(values: Any, dtype: Any = np.float32) -> np.ndarray:
    """Convert tensor-like outputs to numpy arrays.

    Handles PyTorch tensors (detaching, moving to CPU) and generic array-likes.

    Args:
        values: Tensor-like object (PyTorch tensor, numpy array, list, etc.)
        dtype: Target numpy dtype for the output array

    Returns:
        Numpy array with the specified dtype
    """
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy(), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def extract_predictions_array(outputs: Any) -> np.ndarray:
    """Normalize model outputs into a numpy prediction array.

    Handles various output formats from different TSFM models:
    - Tuple outputs (extracts first element)
    - Objects with `quantile_predictions` or `prediction_outputs` attributes
    - Dict outputs with `quantile_predictions` or `prediction_outputs` keys
    - Direct array/tensor outputs

    Args:
        outputs: Raw model output in various formats

    Returns:
        Numpy array containing predictions

    Raises:
        ValueError: If no predictions can be extracted from outputs
    """
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
        raise ValueError("Model output did not include forecast predictions.")

    return tensor_to_numpy(predictions)


def resolve_quantile_index(
    target_q: float,
    quantile_levels: Sequence[float] | None,
    tensor_size: int,
    fallback_mapping: dict[int, Callable[[float], int]] | None = None,
) -> int | None:
    """Resolve quantile index in forecast tensor.

    Tries exact match against quantile_levels first, then applies fallback mapping
    if provided, finally defaults to selecting index closest to median (0.5).

    Args:
        target_q: Target quantile level (e.g., 0.5 for median)
        quantile_levels: Available quantile levels from model config
        tensor_size: Size of the quantile dimension in output tensor
        fallback_mapping: Optional mapping from tensor sizes to index resolver functions

    Returns:
        Index in tensor for the target quantile, or None if cannot be resolved
    """
    # 1. Try exact match against model quantile levels
    if quantile_levels:
        for i, q in enumerate(quantile_levels):
            if np.isclose(q, target_q):
                if tensor_size == len(quantile_levels):
                    return i
                if tensor_size == len(quantile_levels) + 1:
                    return i + 1

    # 2. Apply fallback mapping if provided
    if fallback_mapping and tensor_size in fallback_mapping:
        return fallback_mapping[tensor_size](target_q)

    # 3. Default: select index closest to median (0.5)
    if tensor_size > 0:
        return min(range(tensor_size), key=lambda i: abs(i / max(1, tensor_size - 1) - 0.5))

    return None


def select_median_index(quantile_levels: Sequence[float] | None, size: int) -> int:
    """Select index closest to median (0.5) based on available quantile levels.

    Args:
        quantile_levels: Available quantile levels
        size: Size of the quantile dimension

    Returns:
        Index closest to median
    """
    if not quantile_levels:
        return size // 2
    return min(range(size), key=lambda i: abs(float(quantile_levels[i]) - 0.5))


def extract_point_forecast(
    outputs: Any,
    h: int,
    quantile_levels: list[float] | None = None,
) -> np.ndarray:
    """Extract point forecasts (median) from model outputs.

    Handles 1D, 2D, and 3D output arrays with various quantile layouts.

    Args:
        outputs: Raw model output or pre-extracted prediction array
        h: Forecast horizon
        quantile_levels: Available quantile levels from model config

    Returns:
        1D numpy array of point forecasts with length h
    """
    # Handle both raw outputs and pre-extracted arrays
    arr = outputs if isinstance(outputs, np.ndarray) else extract_predictions_array(outputs)

    quantile_count = len(quantile_levels) if quantile_levels is not None else None

    if arr.ndim == 1:
        values = arr[:h]
    elif arr.ndim == 2:
        if quantile_count is not None and arr.shape[0] == quantile_count:
            q_idx = select_median_index(quantile_levels, arr.shape[0])
            values = arr[q_idx, :h]
        elif quantile_count is not None and arr.shape[1] == quantile_count:
            q_idx = select_median_index(quantile_levels, arr.shape[1])
            values = arr[:h, q_idx]
        else:
            if arr.shape[0] == 1:
                values = arr[0, :h]
            elif arr.shape[1] == 1:
                values = arr[:h, 0]
            else:
                values = arr[0, :h]
    elif arr.ndim == 3:
        # Expected shape is (batch, quantile, horizon), but we keep
        # compatibility for horizon-first tensors.
        sample = arr[0]
        if quantile_count is not None and sample.shape[0] == quantile_count:
            q_idx = select_median_index(quantile_levels, sample.shape[0])
            values = sample[q_idx, :h]
        elif quantile_count is not None and sample.shape[1] == quantile_count:
            q_idx = select_median_index(quantile_levels, sample.shape[1])
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
        raise ValueError(f"Unexpected output rank: {arr.ndim}")

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(h, dtype=np.float32)
    if len(values) < h:
        return np.pad(values, (0, h - len(values)), mode="edge")
    return values[:h]


def extract_batch_forecasts(
    outputs: Any,
    h: int,
    batch_size: int,
    quantile_levels: list[float] | None = None,
) -> list[np.ndarray]:
    """Extract forecast values for each series in a batch from model outputs.

    Handles various output formats from batched inference including:
    - 1D arrays (split equally among batch)
    - 2D arrays (batch, horizon) or (batch, quantile)
    - 3D arrays (batch, quantile, horizon) or (batch, horizon, quantile)

    Args:
        outputs: Raw model output
        h: Forecast horizon
        batch_size: Number of series in the batch
        quantile_levels: Available quantile levels from model config

    Returns:
        List of 1D forecast arrays, one per series in batch
    """
    arr = extract_predictions_array(outputs)
    quantile_count = len(quantile_levels) if quantile_levels is not None else None

    # Handle edge case: model returns single output for entire batch (e.g., test mocks)
    if arr.ndim >= 3 and arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)

    results = []
    for b in range(batch_size):
        if arr.ndim == 1:
            # 1D array - split equally among batch
            chunk_size = len(arr) // batch_size
            values = arr[b * chunk_size : b * chunk_size + h]
        elif arr.ndim == 2:
            # Expected: (batch, horizon) or (batch, quantile)
            if arr.shape[0] == batch_size:
                values = arr[b, :h]
            else:
                # Try to interpret as batched quantiles
                if quantile_count is not None and arr.shape[1] == quantile_count:
                    # (batch, quantile) - single horizon point
                    q_idx = select_median_index(quantile_levels, arr.shape[1])
                    values = np.full(h, arr[b, q_idx], dtype=np.float32)
                else:
                    values = arr[b, :h]
        elif arr.ndim == 3:
            # Expected: (batch, quantile, horizon) or (batch, horizon, quantile)
            sample = arr[b]
            if quantile_count is not None and sample.shape[0] == quantile_count:
                # (quantile, horizon)
                q_idx = select_median_index(quantile_levels, sample.shape[0])
                values = sample[q_idx, :h]
            elif quantile_count is not None and sample.shape[1] == quantile_count:
                # (horizon, quantile)
                q_idx = select_median_index(quantile_levels, sample.shape[1])
                values = sample[:h, q_idx]
            else:
                # Fallback: assume (something, horizon) and take middle
                if sample.shape[0] <= sample.shape[1]:
                    q_idx = sample.shape[0] // 2
                    values = sample[q_idx, :h]
                else:
                    q_idx = sample.shape[1] // 2
                    values = sample[:h, q_idx]
        else:
            raise ValueError(f"Unexpected batch output rank: {arr.ndim}")

        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            values = np.zeros(h, dtype=np.float32)
        elif len(values) < h:
            values = np.pad(values, (0, h - len(values)), mode="edge")
        else:
            values = values[:h]

        results.append(values)

    return results


def _extract_single_quantile_values(
    sample: np.ndarray,
    h: int,
    q_idx: int,
    quantile_count: int,
) -> np.ndarray | None:
    """Extract one quantile trajectory from a single-sample tensor.

    Internal helper for extract_batch_quantiles.

    Args:
        sample: Single sample array (1D or 2D)
        h: Forecast horizon
        q_idx: Quantile index to extract
        quantile_count: Total number of quantiles

    Returns:
        1D array of quantile values or None if cannot extract
    """
    if sample.ndim == 1:
        if quantile_count == 1 and q_idx == 0:
            return sample[:h]
        return None

    if sample.ndim != 2:
        return None

    # (quantile, horizon)
    if sample.shape[0] == quantile_count and q_idx < sample.shape[0]:
        return sample[q_idx, :h]
    # (horizon, quantile)
    if sample.shape[1] == quantile_count and q_idx < sample.shape[1]:
        return sample[:h, q_idx]
    return None


def extract_batch_quantiles(
    outputs: Any,
    h: int,
    batch_size: int,
    quantile_levels: list[float] | None = None,
) -> dict[float, list[np.ndarray]]:
    """Extract requested quantile trajectories for each series in a batch.

    Args:
        outputs: Raw model output
        h: Forecast horizon
        batch_size: Number of series in the batch
        quantile_levels: Quantile levels to extract (e.g., [0.1, 0.5, 0.9])

    Returns:
        Dict mapping quantile level to list of forecast arrays (one per series)
        Only includes quantiles with full batch coverage.
    """
    if not quantile_levels:
        return {}

    arr = extract_predictions_array(outputs)
    quantile_count = len(quantile_levels)

    # Handle edge case: model returns single output for entire batch (e.g., test mocks)
    if arr.ndim >= 3 and arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)

    results: dict[float, list[np.ndarray]] = {q: [] for q in quantile_levels}

    for b in range(batch_size):
        if arr.ndim == 1:
            sample = arr
        elif arr.ndim == 2 and arr.shape[0] == batch_size:
            sample = arr[b]
        elif arr.ndim == 2 and batch_size == 1:
            sample = arr
        elif arr.ndim == 3:
            sample = arr[b]
        else:
            sample = arr

        for q_idx, q in enumerate(quantile_levels):
            values = _extract_single_quantile_values(
                sample=np.asarray(sample, dtype=np.float32),
                h=h,
                q_idx=q_idx,
                quantile_count=quantile_count,
            )
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float32)
            if values.size == 0:
                values = np.zeros(h, dtype=np.float32)
            elif len(values) < h:
                values = np.pad(values, (0, h - len(values)), mode="edge")
            else:
                values = values[:h]
            results[q].append(values)

    # Keep only quantiles with full batch coverage.
    return {q: vals for q, vals in results.items() if len(vals) == batch_size}
