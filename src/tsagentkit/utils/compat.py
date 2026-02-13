"""Compatibility utilities for common patterns.

This module provides helper functions for common patterns that appear
throughout the codebase, reducing duplication and improving consistency.
"""

from __future__ import annotations

import inspect
from typing import Any


def safe_model_dump(obj: Any, default: Any = None) -> Any:
    """Safely dump a Pydantic model or return default.

    This eliminates the repetitive pattern:
        obj.model_dump() if hasattr(obj, "model_dump") else default

    Args:
        obj: Object that may be a Pydantic model with model_dump method
        default: Value to return if obj doesn't have model_dump

    Returns:
        Model dump dict or default value

    Example:
        >>> spec = TaskSpec(...)
        >>> dump = safe_model_dump(spec, {})
        >>> # Equivalent to:
        >>> # dump = spec.model_dump() if hasattr(spec, "model_dump") else {}
    """
    return obj.model_dump() if hasattr(obj, "model_dump") else default


def call_with_optional_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function with only supported keyword arguments.

    Uses inspect.signature to filter kwargs to only those accepted by func.
    Falls back to passing all kwargs if signature inspection fails.

    Args:
        func: Function or callable to invoke
        *args: Positional arguments to pass
        **kwargs: Keyword arguments to filter and pass

    Returns:
        Result of calling func with filtered arguments

    Example:
        >>> def foo(a, b, c=1): ...
        >>> call_with_optional_kwargs(foo, 1, 2, c=3, d=4)  # d is filtered out
    """
    if not kwargs:
        return func(*args)

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        # Fall back to direct call when signature introspection is unsupported
        return func(*args, **kwargs)

    accepted = {k: v for k, v in kwargs.items() if k in params}
    return func(*args, **accepted)
