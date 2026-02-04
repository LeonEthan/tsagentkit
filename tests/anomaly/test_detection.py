"""Tests for anomaly detection."""

import pandas as pd
import pytest

from tsagentkit.anomaly import detect_anomalies


def test_warns_when_uncalibrated_non_strict() -> None:
    """Non-strict mode should warn when calibration is missing."""
    df = pd.DataFrame({
        "unique_id": ["A", "A"],
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "y": [10.0, 12.0],
        "yhat": [11.0, 11.5],
        "yhat_lo_95": [8.0, 9.0],
        "yhat_hi_95": [14.0, 15.0],
    })

    with pytest.warns(UserWarning):
        report = detect_anomalies(
            df,
            method="conformal_interval",
            level=95,
            strict=False,
            calibrator=None,
        )

    assert report is not None
