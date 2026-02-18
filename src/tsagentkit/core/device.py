"""Device resolution utilities for TSFM inference.

Provides automatic device selection (CUDA → MPS → CPU) with override support.
"""

from __future__ import annotations

from typing import Literal

DeviceLiteral = Literal["auto", "cuda", "mps", "cpu"]


def resolve_device(preference: str = "auto", allow_mps: bool = True) -> str:
    """Resolve device for model inference.

    Priority order when preference="auto":
        1. CUDA (if available)
        2. MPS (if available and allow_mps=True)
        3. CPU (fallback)

    Args:
        preference: Device preference ('auto', 'cuda', 'mps', 'cpu')
        allow_mps: Whether to consider MPS as an option (some models
            have instability issues on MPS)

    Returns:
        Resolved device string ('cuda', 'mps', or 'cpu')

    Examples:
        >>> resolve_device("auto")  # Returns 'cuda', 'mps', or 'cpu'
        >>> resolve_device("cuda")  # Returns 'cuda' or 'cpu' if unavailable
        >>> resolve_device("auto", allow_mps=False)  # Skips MPS
    """
    # Validate input
    valid_prefs = ("auto", "cuda", "mps", "cpu")
    if preference not in valid_prefs:
        preference = "auto"

    # Handle explicit device requests
    if preference == "cuda":
        return "cuda" if _cuda_available() else "cpu"

    if preference == "mps":
        return "mps" if _mps_available() else "cpu"

    if preference == "cpu":
        return "cpu"

    # Auto-detect with priority: CUDA > MPS > CPU
    if _cuda_available():
        return "cuda"

    if allow_mps and _mps_available():
        return "mps"

    return "cpu"


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
    except ImportError:
        return False

    cuda = getattr(torch, "cuda", None)
    return bool(cuda is not None and hasattr(cuda, "is_available") and cuda.is_available())


def _mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    try:
        import torch
    except ImportError:
        return False

    backend_mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(
        backend_mps is not None and hasattr(backend_mps, "is_available") and backend_mps.is_available()
    )


def device_to_device_map(device: str) -> str | dict[str, str]:
    """Convert device string to HuggingFace device_map format.

    Args:
        device: Device string ('cuda', 'mps', 'cpu')

    Returns:
        Device map compatible with HF transformers
    """
    # HF transformers typically accepts:
    # - "auto": let HF decide
    # - "cuda", "cpu": simple device string
    # - dict: for multi-GPU sharding
    if device == "cuda":
        return "cuda"
    if device == "mps":
        # Some HF models don't support mps directly, fallback to auto
        return "auto"
    return device
