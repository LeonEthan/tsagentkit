"""Tests for serving refinement helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit.contracts import (
    AnomalySpec,
    CalibratorSpec,
    EAnomalyFail,
    ECalibrationFail,
    PanelContract,
)
from tsagentkit.serving.refinement import (
    calibrate_forecast,
    detect_data_drift,
    detect_forecast_anomalies,
)


def test_calibrate_forecast_uses_cv_wrapper_df() -> None:
    """Calibration should accept cv frames wrapped with a .df attribute."""

    class CVWrapper:
        def __init__(self, df: pd.DataFrame) -> None:
            self.df = df

    forecast = pd.DataFrame(
        {
            "unique_id": ["s1", "s2"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "yhat": [10.0, 20.0],
        }
    )
    cv = pd.DataFrame(
        {
            "unique_id": ["s1", "s2"],
            "y": [9.0, 22.0],
            "yhat": [10.0, 20.0],
        }
    )

    calibrated, artifact = calibrate_forecast(
        forecast=forecast,
        cv_frame=CVWrapper(cv),
        calibrator_spec=CalibratorSpec(method="conformal", level=95, by="unique_id"),
    )

    assert artifact.method == "conformal"
    assert "yhat_lo_95" in calibrated.columns
    assert "yhat_hi_95" in calibrated.columns


def test_calibrate_forecast_requires_cv_residuals() -> None:
    """Calibration should fail fast when cv residuals are unavailable."""
    forecast = pd.DataFrame({"unique_id": ["s1"], "ds": pd.to_datetime(["2024-01-01"]), "yhat": [1.0]})

    with pytest.raises(ECalibrationFail, match="Calibration requires CV residuals from backtest"):
        calibrate_forecast(
            forecast=forecast,
            cv_frame=None,
            calibrator_spec=CalibratorSpec(method="conformal"),
        )


def test_detect_forecast_anomalies_non_strict_without_actuals_returns_none() -> None:
    """Non-strict anomaly detection should no-op when joined actuals are missing."""
    forecast = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2024-01-10"]),
            "yhat": [10.0],
            "yhat_lo_95": [9.0],
            "yhat_hi_95": [11.0],
        }
    )
    history = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [10.0],
        }
    )

    report = detect_forecast_anomalies(
        forecast=forecast,
        historical_data=history,
        panel_contract=PanelContract(),
        anomaly_spec=AnomalySpec(method="conformal", level=95),
        strict=False,
    )

    assert report is None


def test_detect_forecast_anomalies_strict_without_actuals_raises() -> None:
    """Strict anomaly detection should fail when no joined actuals exist."""
    forecast = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2024-01-10"]),
            "yhat": [10.0],
            "yhat_lo_95": [9.0],
            "yhat_hi_95": [11.0],
        }
    )
    history = pd.DataFrame(
        {
            "unique_id": ["s1"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "y": [10.0],
        }
    )

    with pytest.raises(EAnomalyFail, match="No actuals available for anomaly detection"):
        detect_forecast_anomalies(
            forecast=forecast,
            historical_data=history,
            panel_contract=PanelContract(),
            anomaly_spec=AnomalySpec(method="conformal", level=95),
            strict=True,
        )


def test_detect_data_drift_runs_default_psi() -> None:
    """Drift helper should produce a report for numeric features."""
    reference = pd.DataFrame(
        {
            "unique_id": ["s1"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [float(i) for i in range(20)],
            "feature_a": [float(i) for i in range(20)],
        }
    )
    current = reference.copy()

    report = detect_data_drift(reference_data=reference, current_data=current)

    assert report.threshold_used == pytest.approx(0.2)
    assert report.overall_drift_score == pytest.approx(0.0, abs=1e-6)
    assert not report.drift_detected
