"""Tests for tsagentkit.calibration – fit_calibrator, apply_calibrator, CalibratorArtifact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit.calibration import (
    CalibratorArtifact,
    apply_calibrator,
    fit_calibrator,
)
from tsagentkit.contracts import ECalibrationFail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cv_df(n_series: int = 2, n_steps: int = 20, seed: int = 42) -> pd.DataFrame:
    """Deterministic cross-validation DataFrame."""
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    for uid_idx in range(n_series):
        uid = f"series_{uid_idx}"
        for step in range(n_steps):
            y = 10.0 + step
            yhat = y + rng.normal(0, 1.0)
            rows.append({"unique_id": uid, "y": y, "yhat": yhat})
    return pd.DataFrame(rows)


def _make_forecast_df(n_series: int = 2, n_steps: int = 5, seed: int = 42) -> pd.DataFrame:
    """Deterministic forecast DataFrame."""
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    dates = pd.date_range("2024-06-01", periods=n_steps, freq="D")
    for uid_idx in range(n_series):
        uid = f"series_{uid_idx}"
        for ds in dates:
            rows.append(
                {
                    "unique_id": uid,
                    "ds": ds,
                    "yhat": 10.0 + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fit_calibrator
# ---------------------------------------------------------------------------


class TestFitCalibrator:
    def test_conformal_per_series(self) -> None:
        cv = _make_cv_df()
        art = fit_calibrator(cv, method="conformal", level=95, by="unique_id")
        assert art.method == "conformal"
        assert art.level == 95
        assert art.by == "unique_id"
        # Should have one delta per series
        assert "series_0" in art.deltas
        assert "series_1" in art.deltas
        # Deltas should be positive
        for v in art.deltas.values():
            assert v > 0.0

    def test_conformal_global(self) -> None:
        cv = _make_cv_df()
        art = fit_calibrator(cv, method="conformal", level=99, by="global")
        assert art.by == "global"
        assert "global" in art.deltas
        assert art.deltas["global"] > 0.0
        assert art.metadata["n_residuals"] > 0

    def test_none_method(self) -> None:
        cv = _make_cv_df()
        art = fit_calibrator(cv, method="none", level=99)
        assert art.method == "none"
        assert art.deltas == {"global": 0.0}

    def test_empty_cv_raises(self) -> None:
        empty = pd.DataFrame(columns=["unique_id", "y", "yhat"])
        with pytest.raises(ECalibrationFail, match="empty"):
            fit_calibrator(empty, method="conformal")

    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame({"unique_id": ["A"], "x": [1.0]})
        with pytest.raises(ECalibrationFail, match="columns"):
            fit_calibrator(bad, method="conformal")


# ---------------------------------------------------------------------------
# apply_calibrator
# ---------------------------------------------------------------------------


class TestApplyCalibrator:
    def test_adjusts_yhat_lo_hi(self) -> None:
        cv = _make_cv_df()
        art = fit_calibrator(cv, method="conformal", level=95, by="global")
        forecast = _make_forecast_df()
        adjusted = apply_calibrator(forecast, art)
        assert "yhat_lo_95" in adjusted.columns
        assert "yhat_hi_95" in adjusted.columns
        # lo < yhat < hi
        assert (adjusted["yhat_lo_95"] <= adjusted["yhat"]).all()
        assert (adjusted["yhat_hi_95"] >= adjusted["yhat"]).all()

    def test_none_method_passthrough(self) -> None:
        art = CalibratorArtifact(method="none", level=99)
        forecast = _make_forecast_df()
        result = apply_calibrator(forecast, art)
        # Should be unchanged
        assert "yhat_lo_99" not in result.columns

    def test_empty_forecast_returns_empty(self) -> None:
        art = CalibratorArtifact(method="conformal", level=95, deltas={"global": 1.0})
        empty = pd.DataFrame(columns=["unique_id", "ds", "yhat"])
        result = apply_calibrator(empty, art)
        assert result.empty


# ---------------------------------------------------------------------------
# CalibratorArtifact
# ---------------------------------------------------------------------------


class TestCalibratorArtifact:
    def test_get_delta_per_series(self) -> None:
        art = CalibratorArtifact(
            method="conformal",
            level=95,
            by="unique_id",
            deltas={"A": 1.5, "B": 2.0},
        )
        assert art.get_delta("A") == 1.5
        assert art.get_delta("B") == 2.0
        # Unknown series returns 0.0
        assert art.get_delta("C") == 0.0

    def test_get_delta_global(self) -> None:
        art = CalibratorArtifact(
            method="conformal",
            level=99,
            by="global",
            deltas={"global": 3.0},
        )
        assert art.get_delta("any_id") == 3.0
        assert art.get_delta() == 3.0

    def test_frozen(self) -> None:
        art = CalibratorArtifact(method="none", level=99)
        with pytest.raises(AttributeError):
            art.method = "conformal"  # type: ignore[misc]
