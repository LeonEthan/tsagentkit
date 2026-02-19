"""Length limit utilities for TSFM models.

Provides centralized handling of context/prediction length constraints
with support for padding, truncation, and validation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tsagentkit.models.registry import ModelSpec


@dataclass(frozen=True)
class LengthAdjustment:
    """Result of adjusting series length to model limits."""

    data: np.ndarray
    was_padded: bool
    was_truncated: bool
    original_length: int
    adjusted_length: int
    padding_amount: int
    truncation_amount: int


def adjust_context_length(
    context: np.ndarray,
    spec: ModelSpec,
    pad_value: float | None = None,
) -> LengthAdjustment:
    """Adjust context array to meet model length requirements.

    Applies padding and/or truncation based on ModelSpec limits.
    Uses left-padding to preserve recency (most recent values at end).
    Uses left-truncation to preserve recency (oldest values removed).

    Args:
        context: Input time series values
        spec: Model specification with length limits
        pad_value: Value for padding (None = use context mean, 0.0 if empty)

    Returns:
        LengthAdjustment with adjusted data and metadata
    """
    original_length = len(context)
    data = context.copy()
    was_padded = False
    was_truncated = False
    padding_amount = 0
    truncation_amount = 0

    # Determine effective limits
    min_len = spec.min_context_length
    max_len = spec.max_context_length

    # Handle minimum length (padding)
    if min_len is not None and len(data) < min_len:
        if spec.pad_to_min_context:
            pad_amount = min_len - len(data)
            fill = (
                pad_value if pad_value is not None else (float(np.mean(data)) if data.size else 0.0)
            )
            data = np.pad(data, (pad_amount, 0), mode="constant", constant_values=fill)
            was_padded = True
            padding_amount = pad_amount
        else:
            warnings.warn(
                f"Model {spec.name} recommends context >= {min_len}, "
                f"got {len(data)}. Proceeding without padding.",
                stacklevel=2,
            )

    # Handle maximum length (truncation)
    if max_len is not None and len(data) > max_len:
        if spec.truncate_to_max_context:
            # Left-truncate: keep the most recent values (end of array)
            truncation_amount = len(data) - max_len
            data = data[-max_len:]
            was_truncated = True
        else:
            raise ValueError(
                f"Model {spec.name} max context length is {max_len}, "
                f"got {len(data)}. Set truncate_to_max_context=True to auto-truncate."
            )

    return LengthAdjustment(
        data=data,
        was_padded=was_padded,
        was_truncated=was_truncated,
        original_length=original_length,
        adjusted_length=len(data),
        padding_amount=padding_amount,
        truncation_amount=truncation_amount,
    )


def validate_prediction_length(
    h: int,
    spec: ModelSpec,
    strict: bool = False,
) -> int:
    """Validate and optionally clip prediction length.

    Args:
        h: Requested forecast horizon
        spec: Model specification
        strict: If True, raise error on limit violation; if False, warn/clip

    Returns:
        Validated (potentially clipped) horizon

    Raises:
        ValueError: If h exceeds max and strict mode is enabled
    """
    max_h = spec.max_prediction_length

    if max_h is None:
        return h

    if h > max_h:
        if strict:
            raise ValueError(
                f"Model {spec.name} max prediction length is {max_h}, "
                f"requested {h}. Set strict=False to clip automatically."
            )
        warnings.warn(
            f"Model {spec.name} max prediction length is {max_h}, "
            f"requested {h}. Clipping to {max_h}.",
            stacklevel=2,
        )
        return max_h

    return h


def get_effective_limits(
    spec: ModelSpec,
    loaded_model: Any | None = None,
) -> dict[str, int | None]:
    """Get effective length limits for a model.

    For models with dynamic limits (like PatchTST-FM), resolves
    limits from the loaded model instance.

    Args:
        spec: Model specification
        loaded_model: Optional loaded model instance for dynamic resolution

    Returns:
        Dictionary with min_context, max_context, max_prediction
    """
    limits = {
        "min_context": spec.min_context_length,
        "max_context": spec.max_context_length,
        "max_prediction": spec.max_prediction_length,
    }

    # Dynamic resolution for models that need it
    if loaded_model is not None and spec.name == "patchtst_fm":
        config = getattr(loaded_model, "config", None)
        if config:
            ctx = getattr(config, "context_length", None)
            if ctx:
                limits["max_context"] = ctx
                limits["min_context"] = ctx  # PatchTST expects exact length

    return limits


def check_data_compatibility(
    spec: ModelSpec,
    context_length: int,
    prediction_length: int,
) -> dict[str, Any]:
    """Check if model can handle given context and prediction lengths.

    Args:
        spec: Model specification
        context_length: Length of input context
        prediction_length: Forecast horizon

    Returns:
        Dictionary with compatibility info:
        - compatible: bool indicating if model can handle the data
        - issues: list of issues found
        - recommendations: list of recommended actions
    """
    issues: list[str] = []
    recommendations: list[str] = []
    compatible = True

    # Check prediction length
    if spec.max_prediction_length is not None and prediction_length > spec.max_prediction_length:
        issues.append(
            f"Prediction length {prediction_length} exceeds max {spec.max_prediction_length}"
        )
        recommendations.append(
            f"Reduce horizon to {spec.max_prediction_length} or use AR generation"
        )
        compatible = False

    # Check minimum context
    if spec.min_context_length is not None and context_length < spec.min_context_length:
        if spec.pad_to_min_context:
            recommendations.append(
                f"Will pad context from {context_length} to {spec.min_context_length}"
            )
        else:
            issues.append(
                f"Context length {context_length} below minimum {spec.min_context_length}"
            )
            compatible = False

    # Check maximum context
    if spec.max_context_length is not None and context_length > spec.max_context_length:
        if spec.truncate_to_max_context:
            recommendations.append(
                f"Will truncate context from {context_length} to {spec.max_context_length}"
            )
        else:
            issues.append(
                f"Context length {context_length} exceeds max {spec.max_context_length}"
            )
            compatible = False

    return {
        "compatible": compatible,
        "issues": issues,
        "recommendations": recommendations,
    }


__all__ = [
    "LengthAdjustment",
    "adjust_context_length",
    "validate_prediction_length",
    "get_effective_limits",
    "check_data_compatibility",
]
