"""Signature helpers for provenance hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def compute_data_signature(df: pd.DataFrame) -> str:
    """Compute a hash signature for a DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        SHA-256 hash string (truncated to 16 chars)
    """
    cols = sorted(df.columns)
    data_str = ""

    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            values = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
        else:
            values = df[col].astype(str).tolist()
        data_str += f"{col}:{','.join(values)};"

    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def compute_config_signature(config: dict[str, Any]) -> str:
    """Compute a hash signature for a configuration dict.

    Args:
        config: Configuration dictionary

    Returns:
        SHA-256 hash string (truncated to 16 chars)
    """
    json_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


__all__ = ["compute_data_signature", "compute_config_signature"]
