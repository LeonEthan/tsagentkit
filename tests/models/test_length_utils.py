"""Tests for length limit utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsagentkit.models.length_utils import (
    LengthAdjustment,
    adjust_context_length,
    check_data_compatibility,
    get_effective_limits,
    validate_prediction_length,
)
from tsagentkit.models.registry import ModelSpec


class TestLengthAdjustment:
    """Tests for LengthAdjustment dataclass."""

    def test_length_adjustment_creation(self):
        """Test creating LengthAdjustment instances."""
        data = np.array([1.0, 2.0, 3.0])
        adj = LengthAdjustment(
            data=data,
            was_padded=False,
            was_truncated=False,
            original_length=3,
            adjusted_length=3,
            padding_amount=0,
            truncation_amount=0,
        )
        assert adj.original_length == 3
        assert adj.adjusted_length == 3
        assert not adj.was_padded
        assert not adj.was_truncated


class TestAdjustContextLength:
    """Tests for adjust_context_length function."""

    def test_no_adjustment_needed(self):
        """Test context that fits within limits."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=None,
            max_context_length=100,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        context = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = adjust_context_length(context, spec)

        assert np.array_equal(result.data, context)
        assert not result.was_padded
        assert not result.was_truncated
        assert result.original_length == 5
        assert result.adjusted_length == 5

    def test_padding_to_min_length(self):
        """Test padding when context is below minimum."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=10,
            max_context_length=100,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        context = np.array([1.0, 2.0, 3.0])

        result = adjust_context_length(context, spec)

        assert len(result.data) == 10
        assert result.was_padded
        assert not result.was_truncated
        assert result.padding_amount == 7
        assert result.adjusted_length == 10
        # Check left-padding (most recent values at end)
        assert np.array_equal(result.data[-3:], [1.0, 2.0, 3.0])

    def test_padding_with_custom_value(self):
        """Test padding with custom pad value."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=5,
            max_context_length=100,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        context = np.array([1.0, 2.0])

        result = adjust_context_length(context, spec, pad_value=0.0)

        assert len(result.data) == 5
        assert np.array_equal(result.data[:3], [0.0, 0.0, 0.0])
        assert np.array_equal(result.data[3:], [1.0, 2.0])

    def test_padding_with_mean(self):
        """Test padding with mean when pad_value is None."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=5,
            max_context_length=100,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        context = np.array([4.0, 6.0])  # mean = 5.0

        result = adjust_context_length(context, spec, pad_value=None)

        assert len(result.data) == 5
        # Padded with mean
        assert np.allclose(result.data[:3], 5.0)

    def test_no_padding_when_disabled(self):
        """Test warning when padding is disabled."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=10,
            max_context_length=100,
            pad_to_min_context=False,
            truncate_to_max_context=True,
        )
        context = np.array([1.0, 2.0, 3.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = adjust_context_length(context, spec)

            assert len(w) == 1
            assert "recommends context >= 10" in str(w[0].message)
        assert np.array_equal(result.data, context)
        assert not result.was_padded

    def test_truncation_when_too_long(self):
        """Test truncation when context exceeds maximum."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=None,
            max_context_length=5,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        context = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        result = adjust_context_length(context, spec)

        assert len(result.data) == 5
        assert not result.was_padded
        assert result.was_truncated
        assert result.truncation_amount == 2
        # Left-truncation: keep most recent values (end of array)
        assert np.array_equal(result.data, [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_error_when_truncation_disabled(self):
        """Test error when truncation is disabled and context too long."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=None,
            max_context_length=5,
            pad_to_min_context=True,
            truncate_to_max_context=False,
        )
        context = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        with pytest.raises(ValueError, match="max context length is 5"):
            adjust_context_length(context, spec)

    def test_both_padding_and_truncation(self):
        """Test handling when both adjustments are needed (different contexts)."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=5,
            max_context_length=10,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        # Short context
        short = np.array([1.0, 2.0])
        result_short = adjust_context_length(short, spec)
        assert result_short.was_padded
        assert not result_short.was_truncated

        # Long context
        long = np.array(range(20), dtype=float)
        result_long = adjust_context_length(long, spec)
        assert not result_long.was_padded
        assert result_long.was_truncated

    def test_empty_context(self):
        """Test handling of empty context."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=5,
            max_context_length=100,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )
        context = np.array([], dtype=float)

        result = adjust_context_length(context, spec)

        assert len(result.data) == 5
        assert result.was_padded
        # Empty array uses 0.0 as fill
        assert np.array_equal(result.data, [0.0, 0.0, 0.0, 0.0, 0.0])


class TestValidatePredictionLength:
    """Tests for validate_prediction_length function."""

    def test_within_limit(self):
        """Test horizon within limit passes through."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_prediction_length=100,
        )

        result = validate_prediction_length(50, spec)

        assert result == 50

    def test_no_limit(self):
        """Test any horizon allowed when no limit."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_prediction_length=None,
        )

        result = validate_prediction_length(10000, spec)

        assert result == 10000

    def test_clipping_when_exceeds(self):
        """Test horizon is clipped when exceeding limit."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_prediction_length=30,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_prediction_length(50, spec, strict=False)

            assert len(w) == 1
            assert "max prediction length is 30" in str(w[0].message)
        assert result == 30

    def test_error_in_strict_mode(self):
        """Test error when exceeding limit in strict mode."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_prediction_length=30,
        )

        with pytest.raises(ValueError, match="max prediction length is 30"):
            validate_prediction_length(50, spec, strict=True)


class TestGetEffectiveLimits:
    """Tests for get_effective_limits function."""

    def test_static_limits(self):
        """Test getting static limits from spec."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=10,
            max_context_length=100,
            max_prediction_length=50,
        )

        limits = get_effective_limits(spec)

        assert limits["min_context"] == 10
        assert limits["max_context"] == 100
        assert limits["max_prediction"] == 50

    def test_dynamic_limits_for_patchtst(self):
        """Test dynamic limit resolution for PatchTST-FM."""
        spec = ModelSpec(
            name="patchtst_fm",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
        )

        # Mock model with config
        class MockConfig:
            context_length = 512

        class MockModel:
            config = MockConfig()

        limits = get_effective_limits(spec, MockModel())

        assert limits["min_context"] == 512  # PatchTST expects exact length
        assert limits["max_context"] == 512


class TestCheckDataCompatibility:
    """Tests for check_data_compatibility function."""

    def test_fully_compatible(self):
        """Test when data is fully compatible."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=10,
            max_context_length=100,
            max_prediction_length=30,
            pad_to_min_context=True,
            truncate_to_max_context=True,
        )

        result = check_data_compatibility(spec, context_length=50, prediction_length=20)

        assert result["compatible"] is True
        assert len(result["issues"]) == 0

    def test_prediction_too_long(self):
        """Test when prediction exceeds limit."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_prediction_length=30,
        )

        result = check_data_compatibility(spec, context_length=50, prediction_length=50)

        assert result["compatible"] is False
        assert any("Prediction length" in issue for issue in result["issues"])

    def test_context_too_short_with_padding(self):
        """Test when context is too short but padding is enabled."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=100,
            pad_to_min_context=True,
        )

        result = check_data_compatibility(spec, context_length=50, prediction_length=10)

        assert result["compatible"] is True
        assert any("pad context" in rec for rec in result["recommendations"])

    def test_context_too_short_no_padding(self):
        """Test when context is too short and padding is disabled."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            min_context_length=100,
            pad_to_min_context=False,
        )

        result = check_data_compatibility(spec, context_length=50, prediction_length=10)

        assert result["compatible"] is False
        assert any("below minimum" in issue for issue in result["issues"])

    def test_context_too_long_with_truncation(self):
        """Test when context is too long but truncation is enabled."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_context_length=100,
            truncate_to_max_context=True,
        )

        result = check_data_compatibility(spec, context_length=150, prediction_length=10)

        assert result["compatible"] is True
        assert any("truncate context" in rec for rec in result["recommendations"])

    def test_context_too_long_no_truncation(self):
        """Test when context is too long and truncation is disabled."""
        spec = ModelSpec(
            name="test",
            adapter_path="test",
            config_fields={},
            requires=[],
            is_tsfm=True,
            max_context_length=100,
            truncate_to_max_context=False,
        )

        result = check_data_compatibility(spec, context_length=150, prediction_length=10)

        assert result["compatible"] is False
        assert any("exceeds max" in issue for issue in result["issues"])
