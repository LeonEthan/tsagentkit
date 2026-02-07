"""Tests for tsagentkit.utils.signature – compute_data_signature, compute_config_signature."""

from __future__ import annotations

import pandas as pd

from tsagentkit.utils.signature import compute_config_signature, compute_data_signature


# ---------------------------------------------------------------------------
# compute_data_signature
# ---------------------------------------------------------------------------


class TestComputeDataSignature:
    def test_deterministic(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        sig1 = compute_data_signature(df)
        sig2 = compute_data_signature(df)
        assert sig1 == sig2

    def test_different_data_different_hash(self) -> None:
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        assert compute_data_signature(df1) != compute_data_signature(df2)

    def test_handles_datetime_columns(self) -> None:
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
                "y": [1.0, 2.0, 3.0],
            }
        )
        sig = compute_data_signature(df)
        assert isinstance(sig, str)
        assert len(sig) == 16

    def test_hash_length(self) -> None:
        df = pd.DataFrame({"x": [1]})
        sig = compute_data_signature(df)
        assert len(sig) == 16

    def test_column_order_independent(self) -> None:
        """Signature sorts columns, so column order shouldn't matter."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
        assert compute_data_signature(df1) == compute_data_signature(df2)


# ---------------------------------------------------------------------------
# compute_config_signature
# ---------------------------------------------------------------------------


class TestComputeConfigSignature:
    def test_deterministic(self) -> None:
        cfg = {"model": "SeasonalNaive", "h": 12}
        sig1 = compute_config_signature(cfg)
        sig2 = compute_config_signature(cfg)
        assert sig1 == sig2

    def test_order_independent(self) -> None:
        """sort_keys=True ensures key order doesn't matter."""
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert compute_config_signature(cfg1) == compute_config_signature(cfg2)

    def test_different_config_different_hash(self) -> None:
        cfg1 = {"model": "Naive"}
        cfg2 = {"model": "Croston"}
        assert compute_config_signature(cfg1) != compute_config_signature(cfg2)

    def test_hash_length(self) -> None:
        sig = compute_config_signature({"x": 1})
        assert len(sig) == 16

    def test_nested_dict(self) -> None:
        cfg = {"outer": {"inner": 42}}
        sig = compute_config_signature(cfg)
        assert isinstance(sig, str)
        assert len(sig) == 16
