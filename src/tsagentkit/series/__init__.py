"""Series module for tsagentkit.

Provides time series data structures and operations.
"""

from .alignment import align_timezone, fill_gaps, resample_series
from .dataset import TSDataset, build_dataset
from .sparsity import SparsityClass, SparsityProfile, compute_sparsity_profile
from .validation import normalize_panel_columns, validate_contract

__all__ = [
    # Dataset
    "TSDataset",
    "build_dataset",
    # Sparsity
    "SparsityProfile",
    "SparsityClass",
    "compute_sparsity_profile",
    # Alignment
    "align_timezone",
    "resample_series",
    "fill_gaps",
    # Validation helpers (series layer)
    "validate_contract",
    "normalize_panel_columns",
]
