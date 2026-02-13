"""Tests for tsagentkit.features.tsfeatures – _resolve_feature_fns, _prefix_if_conflict."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tsagentkit.features.tsfeatures import (
    _prefix_if_conflict,
    _resolve_feature_fns,
)

# ---------------------------------------------------------------------------
# _resolve_feature_fns
# ---------------------------------------------------------------------------


class TestResolveFeatureFns:
    def test_known_functions(self) -> None:
        """Should resolve callable attributes from the module."""
        mock_mod = MagicMock()
        mock_mod.stl_features = lambda df, freq: df  # a callable
        mock_mod.entropy = lambda df, freq: df
        fns = _resolve_feature_fns(mock_mod, ["stl_features", "entropy"])
        assert fns is not None
        assert len(fns) == 2

    def test_unknown_function_raises(self) -> None:
        """Requesting a non-existent function should raise ValueError."""
        mock_mod = MagicMock(spec=[])  # no attributes
        mock_mod.configure_mock(**{})
        # getattr on MagicMock(spec=[]) for unknown attr returns None
        with pytest.raises(ValueError, match="Unknown tsfeatures function"):
            _resolve_feature_fns(mock_mod, ["nonexistent_fn"])

    def test_empty_list_returns_none(self) -> None:
        mock_mod = MagicMock()
        result = _resolve_feature_fns(mock_mod, [])
        assert result is None

    def test_non_callable_raises(self) -> None:
        """Attribute exists but is not callable → should raise."""
        mock_mod = MagicMock(spec=[])
        mock_mod.some_attr = "not_callable"
        # Need to make getattr return the string
        with pytest.raises(ValueError, match="Unknown tsfeatures function"):
            _resolve_feature_fns(mock_mod, ["some_attr"])


# ---------------------------------------------------------------------------
# _prefix_if_conflict
# ---------------------------------------------------------------------------


class TestPrefixIfConflict:
    def test_no_conflicts(self) -> None:
        df = pd.DataFrame({"unique_id": ["A"], "trend": [1.0], "entropy": [0.5]})
        result_df, result_cols = _prefix_if_conflict(df, ["trend", "entropy"])
        assert result_cols == ["trend", "entropy"]
        assert "trend" in result_df.columns

    def test_renames_reserved(self) -> None:
        """Columns named 'y', 'ds', 'unique_id' should be prefixed with 'tsf_'."""
        df = pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": ["2024-01-01"],
                "y": [1.0],
                "trend": [2.0],
            }
        )
        feature_cols = ["y", "ds", "unique_id", "trend"]
        result_df, result_cols = _prefix_if_conflict(df, feature_cols)
        assert "tsf_y" in result_cols
        assert "tsf_ds" in result_cols
        assert "tsf_unique_id" in result_cols
        assert "trend" in result_cols
        assert "tsf_y" in result_df.columns

    def test_partial_conflicts(self) -> None:
        df = pd.DataFrame(
            {
                "unique_id": ["A"],
                "y": [1.0],
                "entropy": [0.3],
            }
        )
        feature_cols = ["y", "entropy"]
        result_df, result_cols = _prefix_if_conflict(df, feature_cols)
        assert "tsf_y" in result_cols
        assert "entropy" in result_cols
        assert "tsf_y" in result_df.columns
        assert "entropy" in result_df.columns


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_tsfeatures_import_function_exists(self) -> None:
        """Verify _import_tsfeatures is importable (not that tsfeatures is installed)."""
        from tsagentkit.features.tsfeatures import _import_tsfeatures

        assert callable(_import_tsfeatures)
