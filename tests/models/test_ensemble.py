"""Tests for ensemble utilities."""

import pandas as pd
import pytest

from tsagentkit.models.ensemble import (
    _is_tsfm_model,
    compute_median_ensemble,
)


class TestIsTsfmModel:
    """Tests for _is_tsfm_model helper."""

    def test_tsfm_prefix(self):
        """Should identify tsfm- prefixed models."""
        assert _is_tsfm_model("tsfm-chronos") is True
        assert _is_tsfm_model("tsfm-timesfm") is True
        assert _is_tsfm_model("tsfm-moirai") is True

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert _is_tsfm_model("TSFM-CHRONOS") is True
        assert _is_tsfm_model("Tsfm-TimesFM") is True

    def test_non_tsfm_models(self):
        """Should return False for non-TSFM models."""
        assert _is_tsfm_model("naive") is False
        assert _is_tsfm_model("seasonal_naive") is False
        assert _is_tsfm_model("statsforecast-ets") is False
        assert _is_tsfm_model("sktime-arima") is False


class TestComputeMedianEnsemble:
    """Tests for compute_median_ensemble function."""

    def test_basic_median_computation(self):
        """Should compute median yhat correctly."""
        # Create two simple forecasts
        forecast1 = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "yhat": [10.0, 20.0, 30.0, 40.0],
            "model": "model1",
        })
        forecast2 = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "yhat": [20.0, 30.0, 40.0, 50.0],
            "model": "model2",
        })

        forecasts = [forecast1, forecast2]
        result = compute_median_ensemble(forecasts)

        # Median of [10, 20] = 15, [20, 30] = 25, etc.
        expected = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "yhat": [15.0, 25.0, 35.0, 45.0],
            "model": "median_ensemble",
        })

        pd.testing.assert_frame_equal(
            result.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            expected.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            check_like=True,
        )

    def test_median_with_quantiles(self):
        """Should preserve quantile columns from first forecast."""
        forecast1 = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "yhat": [10.0, 20.0],
            "q0.1": [8.0, 18.0],
            "q0.9": [12.0, 22.0],
            "model": "model1",
        })
        forecast2 = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "yhat": [20.0, 30.0],
            "q0.1": [18.0, 28.0],
            "q0.9": [22.0, 32.0],
            "model": "model2",
        })

        forecasts = [forecast1, forecast2]
        result = compute_median_ensemble(forecasts, first_forecast=forecast1)

        # Check yhat is median
        assert result["yhat"].tolist() == [15.0, 25.0]
        assert result["model"].iloc[0] == "median_ensemble"

        # Check quantiles are preserved from first forecast
        assert "q0.1" in result.columns
        assert "q0.9" in result.columns
        assert result["q0.1"].tolist() == [8.0, 18.0]
        assert result["q0.9"].tolist() == [12.0, 22.0]

    def test_empty_forecasts_raises(self):
        """Should raise ValueError for empty forecasts list."""
        with pytest.raises(ValueError, match="No forecasts provided"):
            compute_median_ensemble([])

    def test_missing_required_columns_raises(self):
        """Should raise error if required columns missing."""
        forecast = pd.DataFrame({
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
            # Missing yhat
            "model": "model1",
        })

        from tsagentkit.contracts import EModelPredictFailed

        with pytest.raises(EModelPredictFailed):
            compute_median_ensemble([forecast])

    def test_three_model_median(self):
        """Should compute median correctly for odd number of models."""
        base = {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2024-01-01"]),
        }

        forecast1 = pd.DataFrame({**base, "yhat": [10.0], "model": "model1"})
        forecast2 = pd.DataFrame({**base, "yhat": [20.0], "model": "model2"})
        forecast3 = pd.DataFrame({**base, "yhat": [30.0], "model": "model3"})

        forecasts = [forecast1, forecast2, forecast3]
        result = compute_median_ensemble(forecasts)

        # Median of [10, 20, 30] = 20
        assert result["yhat"].iloc[0] == 20.0


class TestComputeMedianEnsembleEdgeCases:
    """Edge case tests for compute_median_ensemble."""

    def test_single_forecast(self):
        """Should handle single forecast (median of one is itself)."""
        forecast = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "yhat": [10.0, 20.0],
            "model": "model1",
        })

        result = compute_median_ensemble([forecast])

        assert result["yhat"].tolist() == [10.0, 20.0]
        assert result["model"].iloc[0] == "median_ensemble"

    def test_different_timestamps_per_series(self):
        """Should handle series with different forecast timestamps."""
        forecast1 = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.to_datetime([
                "2024-01-01", "2024-01-02",
                "2024-01-01", "2024-01-02",
            ]),
            "yhat": [10.0, 20.0, 100.0, 200.0],
            "model": "model1",
        })
        forecast2 = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.to_datetime([
                "2024-01-01", "2024-01-02",
                "2024-01-01", "2024-01-02",
            ]),
            "yhat": [20.0, 30.0, 200.0, 300.0],
            "model": "model2",
        })

        result = compute_median_ensemble([forecast1, forecast2])

        # Check each series separately
        result_a = result[result["unique_id"] == "A"].sort_values("ds")
        result_b = result[result["unique_id"] == "B"].sort_values("ds")

        assert result_a["yhat"].tolist() == [15.0, 25.0]
        assert result_b["yhat"].tolist() == [150.0, 250.0]
