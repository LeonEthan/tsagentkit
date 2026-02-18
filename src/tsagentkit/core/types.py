"""Shared type definitions for tsagentkit.

Type aliases used across modules for clarity and consistency.
"""

from __future__ import annotations

from typing import Any, TypeVar

# Model artifact type - adapters decide what to store
ModelArtifact = Any

# Generic type variable
T = TypeVar("T")

# Common return types
DataFrame = Any  # pd.DataFrame - avoid import at type-check time

__all__ = [
    "ModelArtifact",
    "T",
    "DataFrame",
]
