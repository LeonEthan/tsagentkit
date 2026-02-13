"""Post-prediction refinement helpers for serving workflows."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from tsagentkit.contracts import (
    AnomalySpec,
    CalibratorSpec,
    EAnomalyFail,
    ECalibrationFail,
    PanelContract,
)


def calibrate_forecast(
    forecast: pd.DataFrame,
    cv_frame: Any,
    calibrator_spec: CalibratorSpec,
) -> tuple[pd.DataFrame, Any]:
    """Fit and apply a calibration artifact to a forecast frame."""
    if cv_frame is None:
        raise ECalibrationFail("Calibration requires CV residuals from backtest.")

    if hasattr(cv_frame, "df"):
        cv_frame = cv_frame.df

    from tsagentkit.calibration import apply_calibrator, fit_calibrator

    artifact = fit_calibrator(
        cv_frame,
        method=calibrator_spec.method,
        level=calibrator_spec.level,
        by=calibrator_spec.by,
    )
    calibrated = apply_calibrator(forecast, artifact)
    return calibrated, artifact


def detect_forecast_anomalies(
    forecast: pd.DataFrame,
    historical_data: pd.DataFrame,
    panel_contract: PanelContract,
    anomaly_spec: AnomalySpec,
    calibration_artifact: Any | None = None,
    strict: bool = False,
) -> Any | None:
    """Run anomaly detection when joined actuals are available."""
    actuals = historical_data[
        [
            panel_contract.unique_id_col,
            panel_contract.ds_col,
            panel_contract.y_col,
        ]
    ].copy()
    merged = forecast.merge(
        actuals,
        on=[panel_contract.unique_id_col, panel_contract.ds_col],
        how="left",
    )

    if not merged[panel_contract.y_col].notna().any():
        if strict:
            raise EAnomalyFail("No actuals available for anomaly detection.")
        return None

    from tsagentkit.anomaly import detect_anomalies

    return detect_anomalies(
        merged,
        method=anomaly_spec.method,
        level=anomaly_spec.level,
        score=anomaly_spec.score,
        calibrator=calibration_artifact,
        strict=strict,
    )


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    method: Literal["psi", "ks"] = "psi",
    threshold: float | None = None,
) -> Any:
    """Detect distribution drift between reference and current data."""
    from tsagentkit.monitoring import DriftDetector

    detector = DriftDetector(method=method, threshold=threshold)
    return detector.detect(
        reference_data=reference_data,
        current_data=current_data,
    )

