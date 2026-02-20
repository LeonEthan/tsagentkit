"""Tests for output_utils module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tsagentkit.models.output_utils import (
    extract_batch_forecasts,
    extract_batch_quantiles,
    extract_point_forecast,
    extract_predictions_array,
    resolve_quantile_index,
    select_median_index,
    tensor_to_numpy,
)


class TestTensorToNumpy:
    """Tests for tensor_to_numpy function."""

    def test_numpy_array_input(self) -> None:
        """Should return numpy array as-is with correct dtype."""
        arr = np.array([1.0, 2.0, 3.0])
        result = tensor_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr)

    def test_list_input(self) -> None:
        """Should convert list to numpy array."""
        result = tensor_to_numpy([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_pytorch_tensor_mock(self) -> None:
        """Should handle PyTorch-like tensor with detach/cpu/numpy methods."""
        tensor_mock = MagicMock()
        tensor_mock.detach.return_value = tensor_mock
        tensor_mock.cpu.return_value = tensor_mock
        tensor_mock.numpy.return_value = np.array([1.0, 2.0, 3.0])

        result = tensor_to_numpy(tensor_mock)
        assert isinstance(result, np.ndarray)
        tensor_mock.detach.assert_called_once()
        tensor_mock.cpu.assert_called_once()
        tensor_mock.numpy.assert_called_once()

    def test_custom_dtype(self) -> None:
        """Should respect custom dtype parameter."""
        arr = np.array([1.0, 2.0, 3.0])
        result = tensor_to_numpy(arr, dtype=np.float64)
        assert result.dtype == np.float64


class MockOutput:
    """Mock output object for testing."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestExtractPredictionsArray:
    """Tests for extract_predictions_array function."""

    def test_tuple_output(self) -> None:
        """Should extract first element from tuple."""
        arr = np.array([1.0, 2.0, 3.0])
        result = extract_predictions_array((arr,))
        np.testing.assert_array_equal(result, arr)

    def test_object_with_quantile_predictions(self) -> None:
        """Should extract quantile_predictions attribute."""
        arr = np.array([1.0, 2.0, 3.0])
        output = MockOutput(quantile_predictions=arr)
        result = extract_predictions_array(output)
        np.testing.assert_array_equal(result, arr)

    def test_object_with_prediction_outputs(self) -> None:
        """Should extract prediction_outputs attribute."""
        arr = np.array([1.0, 2.0, 3.0])
        output = MockOutput(prediction_outputs=arr)
        result = extract_predictions_array(output)
        np.testing.assert_array_equal(result, arr)

    def test_dict_with_quantile_predictions(self) -> None:
        """Should extract quantile_predictions from dict."""
        arr = np.array([1.0, 2.0, 3.0])
        result = extract_predictions_array({"quantile_predictions": arr})
        np.testing.assert_array_equal(result, arr)

    def test_dict_with_prediction_outputs(self) -> None:
        """Should extract prediction_outputs from dict."""
        arr = np.array([1.0, 2.0, 3.0])
        result = extract_predictions_array({"prediction_outputs": arr})
        np.testing.assert_array_equal(result, arr)

    def test_direct_array(self) -> None:
        """Should handle direct array input."""
        arr = np.array([1.0, 2.0, 3.0])
        result = extract_predictions_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_none_output_raises(self) -> None:
        """Should raise ValueError for None output."""
        with pytest.raises(ValueError, match="did not include forecast predictions"):
            extract_predictions_array(None)

    def test_empty_dict_raises(self) -> None:
        """Should raise ValueError for empty dict."""
        with pytest.raises(ValueError, match="did not include forecast predictions"):
            extract_predictions_array({})


class TestResolveQuantileIndex:
    """Tests for resolve_quantile_index function."""

    def test_exact_match_same_size(self) -> None:
        """Should return exact index when tensor_size matches quantile_levels."""
        result = resolve_quantile_index(
            target_q=0.5,
            quantile_levels=[0.1, 0.5, 0.9],
            tensor_size=3,
        )
        assert result == 1

    def test_exact_match_plus_one(self) -> None:
        """Should return index+1 when tensor_size is quantile_levels+1."""
        result = resolve_quantile_index(
            target_q=0.5,
            quantile_levels=[0.1, 0.5, 0.9],
            tensor_size=4,
        )
        assert result == 2  # index 1 + 1

    def test_fallback_mapping(self) -> None:
        """Should apply fallback mapping when exact match fails."""
        fallback = {5: lambda q: int(q * 4)}  # Custom mapping for size 5
        result = resolve_quantile_index(
            target_q=0.5,
            quantile_levels=[0.1, 0.9],  # Won't match 0.5
            tensor_size=5,
            fallback_mapping=fallback,
        )
        assert result == 2  # 0.5 * 4 = 2

    def test_default_median_selection(self) -> None:
        """Should default to median selection when no match and no fallback."""
        result = resolve_quantile_index(
            target_q=0.7,
            quantile_levels=None,
            tensor_size=5,
        )
        # For size 5, indices are 0,1,2,3,4. Closest to 0.5 is index 2.
        assert result == 2

    def test_zero_tensor_size(self) -> None:
        """Should return None for zero tensor size."""
        result = resolve_quantile_index(
            target_q=0.5,
            quantile_levels=None,
            tensor_size=0,
        )
        assert result is None


class TestSelectMedianIndex:
    """Tests for select_median_index function."""

    def test_with_quantile_levels(self) -> None:
        """Should select index with quantile closest to 0.5."""
        result = select_median_index([0.1, 0.5, 0.9], 3)
        assert result == 1

    def test_with_closest_quantile(self) -> None:
        """Should select closest to 0.5 when exact 0.5 not present."""
        result = select_median_index([0.1, 0.4, 0.8], 3)
        assert result == 1  # 0.4 is closest to 0.5

    def test_without_quantile_levels(self) -> None:
        """Should return middle index when no quantile levels."""
        result = select_median_index(None, 5)
        assert result == 2  # 5 // 2

    def test_single_element(self) -> None:
        """Should return 0 for single element."""
        result = select_median_index(None, 1)
        assert result == 0


class TestExtractPointForecast:
    """Tests for extract_point_forecast function."""

    def test_1d_array(self) -> None:
        """Should extract first h values from 1D array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_point_forecast(arr, h=3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_2d_quantile_first_axis(self) -> None:
        """Should extract median from 2D array with quantiles on axis 0."""
        # Shape: (3 quantiles, 5 horizon)
        arr = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ])
        result = extract_point_forecast(arr, h=3, quantile_levels=[0.1, 0.5, 0.9])
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])  # median row

    def test_2d_quantile_second_axis(self) -> None:
        """Should extract median from 2D array with quantiles on axis 1."""
        # Shape: (5 horizon, 3 quantiles)
        arr = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ])
        result = extract_point_forecast(arr, h=3, quantile_levels=[0.1, 0.5, 0.9])
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])  # median column

    def test_2d_single_row(self) -> None:
        """Should handle single-row 2D array."""
        arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = extract_point_forecast(arr, h=3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_3d_batch_first(self) -> None:
        """Should extract from first batch of 3D array."""
        # Shape: (2 batch, 3 quantiles, 5 horizon)
        arr = np.array([
            [[1.0, 2.0, 3.0, 4.0, 5.0],
             [2.0, 3.0, 4.0, 5.0, 6.0],
             [3.0, 4.0, 5.0, 6.0, 7.0]],
            [[10.0, 20.0, 30.0, 40.0, 50.0],
             [20.0, 30.0, 40.0, 50.0, 60.0],
             [30.0, 40.0, 50.0, 60.0, 70.0]],
        ])
        result = extract_point_forecast(arr, h=3, quantile_levels=[0.1, 0.5, 0.9])
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])  # first batch, median

    def test_padding_short_output(self) -> None:
        """Should pad with edge value if output shorter than h."""
        arr = np.array([1.0, 2.0])
        result = extract_point_forecast(arr, h=5)
        np.testing.assert_array_equal(result, [1.0, 2.0, 2.0, 2.0, 2.0])

    def test_empty_output_filled_with_zeros(self) -> None:
        """Should return zeros for empty output."""
        arr = np.array([])
        result = extract_point_forecast(arr, h=3)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_invalid_rank_raises(self) -> None:
        """Should raise ValueError for arrays with rank > 3."""
        arr = np.zeros((2, 2, 2, 2))
        with pytest.raises(ValueError, match="Unexpected output rank"):
            extract_point_forecast(arr, h=2)

    def test_horizon_first_quantile_layout(self) -> None:
        """Should extract median from horizon-first layout (horizon, quantile)."""
        # Shape: (1 batch, 7 horizon, 3 quantiles) - horizon-first
        arr = np.array([
            [
                [1, 2, 3],
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
                [41, 42, 43],
                [51, 52, 53],
                [61, 62, 63],
            ]
        ])
        result = extract_point_forecast(arr, h=7, quantile_levels=[0.1, 0.5, 0.9])
        # Should extract median (0.5) which is at column index 1
        np.testing.assert_array_equal(result, [2, 12, 22, 32, 42, 52, 62])


class TestExtractBatchForecasts:
    """Tests for extract_batch_forecasts function."""

    def test_2d_batch_first_axis(self) -> None:
        """Should extract from 2D array with batch on axis 0."""
        # Shape: (3 batch, 5 horizon)
        arr = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [100.0, 200.0, 300.0, 400.0, 500.0],
        ])
        result = extract_batch_forecasts(arr, h=3, batch_size=3)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[1], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result[2], [100.0, 200.0, 300.0])

    def test_3d_batch_quantile_horizon(self) -> None:
        """Should extract from 3D array (batch, quantile, horizon)."""
        # Shape: (2 batch, 3 quantiles, 5 horizon)
        arr = np.array([
            [[1.0, 2.0, 3.0, 4.0, 5.0],
             [2.0, 3.0, 4.0, 5.0, 6.0],
             [3.0, 4.0, 5.0, 6.0, 7.0]],
            [[10.0, 20.0, 30.0, 40.0, 50.0],
             [20.0, 30.0, 40.0, 50.0, 60.0],
             [30.0, 40.0, 50.0, 60.0, 70.0]],
        ])
        result = extract_batch_forecasts(
            arr, h=3, batch_size=2, quantile_levels=[0.1, 0.5, 0.9]
        )
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [2.0, 3.0, 4.0])  # median
        np.testing.assert_array_equal(result[1], [20.0, 30.0, 40.0])  # median

    def test_1d_array_split(self) -> None:
        """Should split 1D array equally among batch."""
        # 6 elements, batch_size=2 -> 3 per batch
        arr = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        result = extract_batch_forecasts(arr, h=2, batch_size=2)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [10.0, 20.0])

    def test_single_output_broadcast(self) -> None:
        """Should broadcast single output to all batch items."""
        # Shape: (1 batch, 5 horizon)
        arr = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        result = extract_batch_forecasts(arr, h=3, batch_size=3)
        assert len(result) == 3
        for r in result:
            np.testing.assert_array_equal(r, [1.0, 2.0, 3.0])

    def test_padding_short_values(self) -> None:
        """Should pad values shorter than h."""
        arr = np.array([[1.0, 2.0]])  # Only 2 values
        result = extract_batch_forecasts(arr, h=5, batch_size=1)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 2.0, 2.0, 2.0])

    def test_empty_values_filled_with_zeros(self) -> None:
        """Should return zeros for empty values."""
        arr = np.array([[]])
        result = extract_batch_forecasts(arr, h=3, batch_size=1)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 0.0])

    def test_invalid_rank_raises(self) -> None:
        """Should raise ValueError for arrays with rank > 3."""
        arr = np.zeros((2, 2, 2, 2))
        with pytest.raises(ValueError, match="Unexpected batch output rank"):
            extract_batch_forecasts(arr, h=2, batch_size=1)


class TestExtractBatchQuantiles:
    """Tests for extract_batch_quantiles function."""

    def test_empty_quantile_levels(self) -> None:
        """Should return empty dict for empty quantile_levels."""
        arr = np.array([1.0, 2.0, 3.0])
        result = extract_batch_quantiles(arr, h=3, batch_size=1, quantile_levels=[])
        assert result == {}

    def test_3d_batch_quantile_horizon(self) -> None:
        """Should extract quantiles from 3D array (batch, quantile, horizon)."""
        # Shape: (2 batch, 3 quantiles, 5 horizon)
        arr = np.array([
            [[1.0, 2.0, 3.0, 4.0, 5.0],
             [2.0, 3.0, 4.0, 5.0, 6.0],
             [3.0, 4.0, 5.0, 6.0, 7.0]],
            [[10.0, 20.0, 30.0, 40.0, 50.0],
             [20.0, 30.0, 40.0, 50.0, 60.0],
             [30.0, 40.0, 50.0, 60.0, 70.0]],
        ])
        result = extract_batch_quantiles(
            arr, h=3, batch_size=2, quantile_levels=[0.1, 0.5, 0.9]
        )
        assert len(result) == 3
        np.testing.assert_array_equal(result[0.1][0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[0.5][0], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result[0.9][0], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result[0.1][1], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result[0.5][1], [20.0, 30.0, 40.0])
        np.testing.assert_array_equal(result[0.9][1], [30.0, 40.0, 50.0])

    def test_2d_batch_no_extraction(self) -> None:
        """2D arrays without proper quantile dimension are not extracted.

        The extract_batch_quantiles function is designed for 3D outputs
        (batch, quantile, horizon). 2D arrays without proper shape
        are filtered out.
        """
        # Shape: (2 batch, 3 quantiles) - insufficient for quantile extraction
        arr = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ])
        result = extract_batch_quantiles(
            arr, h=1, batch_size=2, quantile_levels=[0.1, 0.5, 0.9]
        )
        # 2D arrays are not handled by _extract_single_quantile_values
        assert result == {}

    def test_single_output_broadcast(self) -> None:
        """Should broadcast single output to all batch items."""
        arr = np.array([[[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]]])
        result = extract_batch_quantiles(
            arr, h=3, batch_size=2, quantile_levels=[0.1, 0.5, 0.9]
        )
        assert len(result) == 3
        # Should have 2 entries per quantile (broadcasted)
        assert len(result[0.1]) == 2
        np.testing.assert_array_equal(result[0.1][0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[0.1][1], [1.0, 2.0, 3.0])

    def test_incomplete_batch_filtered(self) -> None:
        """Should filter quantiles without full batch coverage."""
        # If extraction fails for some items, those quantiles are filtered
        # This test uses a shape that won't match quantile extraction
        arr = np.array([[1.0, 2.0]])  # shape (2, 2) with 3 quantiles requested
        result = extract_batch_quantiles(
            arr, h=1, batch_size=2, quantile_levels=[0.1, 0.5, 0.9]
        )
        # Should be empty since shape doesn't match quantile count
        assert result == {}

    def test_padding_short_quantiles(self) -> None:
        """Should pad quantile values shorter than h."""
        arr = np.array([[[1.0, 2.0]]])  # Only 2 values
        result = extract_batch_quantiles(
            arr, h=5, batch_size=1, quantile_levels=[0.5]
        )
        assert len(result) == 1
        np.testing.assert_array_equal(result[0.5][0], [1.0, 2.0, 2.0, 2.0, 2.0])
