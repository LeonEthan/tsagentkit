"""Signature helpers for provenance hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def compute_signature(
    obj: pd.DataFrame | dict[str, Any],
    data_type: str = "auto",
) -> str:
    """Compute a hash signature for data or configuration.

    Args:
        obj: DataFrame or configuration dictionary to hash
        data_type: Type of object ("data", "config", or "auto" for auto-detect)

    Returns:
        SHA-256 hash string (truncated to 16 chars)

    Examples:
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> sig = compute_signature(df, data_type="data")
        >>> len(sig)
        16

        >>> config = {"model": "naive", "horizon": 7}
        >>> sig = compute_signature(config, data_type="config")
        >>> len(sig)
        16
    """
    if data_type == "auto":
        if isinstance(obj, pd.DataFrame):
            data_type = "data"
        else:
            data_type = "config"

    if data_type == "data":
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        cols = sorted(df.columns)
        data_str = ""

        for col in cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                values = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
            else:
                values = df[col].astype(str).tolist()
            data_str += f"{col}:{','.join(values)};"

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    # config type
    config = obj if isinstance(obj, dict) else dict(obj)
    json_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def compute_data_signature(df: pd.DataFrame) -> str:
    """Compute a hash signature for a DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        SHA-256 hash string (truncated to 16 chars)

    Deprecated: Use compute_signature(df, data_type="data") instead.
    """
    return compute_signature(df, data_type="data")


def compute_config_signature(config: dict[str, Any]) -> str:
    """Compute a hash signature for a configuration dict.

    Args:
        config: Configuration dictionary

    Returns:
        SHA-256 hash string (truncated to 16 chars)

    Deprecated: Use compute_signature(config, data_type="config") instead.
    """
    return compute_signature(config, data_type="config")


__all__ = ["compute_signature", "compute_data_signature", "compute_config_signature"]
