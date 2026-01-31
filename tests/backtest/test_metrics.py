"""Tests for backtest/metrics.py."""

import numpy as np
import pytest

from tsagentkit.backtest import (
    compute_all_metrics,
    compute_metrics_by_series,
    mae,
    mase,
    pinball_loss,
    rmse,
    smape,
    wape,
)


class TestWape:
    """Tests for WAPE metric."""

    def test_perfect_forecast(self) -> None:
        """Test WAPE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert wape(y_true, y_pred) == 0.0

    def test_nonzero_error(self) -> None:
        """Test WAPE with forecast error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # Errors: [1, 0, 1] = 2
        # Sum abs true: 6
        # WAPE: 2/6 = 0.333
        assert wape(y_true, y_pred) == pytest.approx(1 / 3)

    def test_zero_sum_raises(self) -> None:
        """Test that WAPE raises when sum of y_true is zero."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="zero"):
            wape(y_true, y_pred)


class TestSmape:
    """Tests for SMAPE metric."""

    def test_perfect_forecast(self) -> None:
        """Test SMAPE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert smape(y_true, y_pred) == 0.0

    def test_typical_case(self) -> None:
        """Test SMAPE with typical values."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        # Error 1: |100-110| / ((100+110)/2) = 10/105 = 0.095
        # Error 2: |200-190| / ((200+190)/2) = 10/195 = 0.051
        # Mean: (0.095 + 0.051) / 2 = 0.073
        result = smape(y_true, y_pred)
        assert result > 0.0
        assert result < 0.1

    def test_handles_zero(self) -> None:
        """Test SMAPE handles zero values."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 2.0])
        # First item: both zero, denominator 0, skipped
        # Second item: |1-2| / ((1+2)/2) = 1/1.5 = 0.667
        result = smape(y_true, y_pred)
        assert result == pytest.approx(2 / 3)


class TestMase:
    """Tests for MASE metric."""

    def test_perfect_forecast(self) -> None:
        """Test MASE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_train = np.array([1.0, 2.0, 3.0, 4.0])
        # Forecast MAE: 0
        # Naive MAE > 0
        # MASE: 0 / naive_mae = 0
        assert mase(y_true, y_pred, y_train) == 0.0

    def test_better_than_naive(self) -> None:
        """Test MASE when forecast is better than naive."""
        y_true = np.array([10.0, 11.0, 12.0])
        y_pred = np.array([10.1, 10.9, 12.1])  # Very close
        y_train = np.array([1.0, 5.0, 1.0, 5.0])  # High naive error
        result = mase(y_true, y_pred, y_train)
        assert result < 1.0  # Better than naive

    def test_worse_than_naive(self) -> None:
        """Test MASE when forecast is worse than naive."""
        y_true = np.array([100.0, 0.0, 100.0])
        y_pred = np.array([0.0, 100.0, 0.0])  # Opposite
        y_train = np.array([1.0, 2.0, 1.0, 2.0])  # Naive error = 1.0
        result = mase(y_true, y_pred, y_train, season_length=1)
        assert result > 1.0  # Worse than naive

    def test_zero_naive_mae_raises(self) -> None:
        """Test that MASE raises when naive MAE is zero."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])
        y_train = np.array([1.0, 1.0, 1.0, 1.0])  # Zero naive error
        with pytest.raises(ValueError, match="zero"):
            mase(y_true, y_pred, y_train)

    def test_seasonal_naive(self) -> None:
        """Test MASE with seasonal naive."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([10.5, 19.5])
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # Season_length=2: errors |3-1|, |4-2|, |5-3|, |6-4| = [2,2,2,2]
        # Naive MAE = 2
        result = mase(y_true, y_pred, y_train, season_length=2)
        assert result > 0


class TestPinballLoss:
    """Tests for pinball loss."""

    def test_median_quantile(self) -> None:
        """Test pinball loss at median (0.5)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_quantile = np.array([1.5, 2.5, 2.5])
        # Over-prediction (q=0.5): 0.5 * error
        # Under-prediction (q=0.5): -0.5 * error
        result = pinball_loss(y_true, y_quantile, 0.5)
        assert result > 0

    def test_high_quantile(self) -> None:
        """Test pinball loss at high quantile (0.9)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_quantile = np.array([2.0, 2.0, 2.0])
        # Heavily penalizes under-prediction
        result = pinball_loss(y_true, y_quantile, 0.9)
        assert result > 0

    def test_low_quantile(self) -> None:
        """Test pinball loss at low quantile (0.1)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_quantile = np.array([2.0, 2.0, 2.0])
        # Heavily penalizes over-prediction
        result = pinball_loss(y_true, y_quantile, 0.1)
        assert result > 0


class TestMae:
    """Tests for MAE metric."""

    def test_perfect(self) -> None:
        """Test MAE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_pred) == 0.0

    def test_typical(self) -> None:
        """Test MAE with typical values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # Errors: 1, 0, 1
        # Mean: 2/3
        assert mae(y_true, y_pred) == pytest.approx(2 / 3)


class TestRmse:
    """Tests for RMSE metric."""

    def test_perfect(self) -> None:
        """Test RMSE with perfect forecast."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_typical(self) -> None:
        """Test RMSE with typical values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # Squared errors: 1, 0, 1
        # Mean: 2/3
        # Root: sqrt(2/3)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2 / 3))


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_returns_all_metrics(self) -> None:
        """Test that all expected metrics are returned."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        metrics = compute_all_metrics(y_true, y_pred, y_train, season_length=1)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "wape" in metrics
        assert "smape" in metrics
        assert "mase" in metrics

    def test_with_quantiles(self) -> None:
        """Test with quantile forecasts."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_quantiles = {
            0.1: np.array([0.8, 1.8, 2.8]),
            0.9: np.array([1.2, 2.2, 3.2]),
        }

        metrics = compute_all_metrics(y_true, y_pred, y_quantiles=y_quantiles)

        assert "pinball_0.10" in metrics
        assert "pinball_0.90" in metrics

    def test_no_train_returns_nan_mase(self) -> None:
        """Test MASE is NaN when no train data provided."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert np.isnan(metrics["mase"])


class TestComputeMetricsBySeries:
    """Tests for compute_metrics_by_series function."""

    def test_multiple_series(self) -> None:
        """Test computing metrics for multiple series."""
        import pandas as pd

        df = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B", "B"],
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "yhat": [1.1, 2.1, 2.9, 11.0, 19.0, 31.0],
        })

        results = compute_metrics_by_series(df)

        assert "A" in results
        assert "B" in results
        assert "mae" in results["A"]
        assert "mae" in results["B"]
