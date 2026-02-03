"""Anomaly detection based on forecast intervals/quantiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from tsagentkit.contracts import EAnomalyFail
from tsagentkit.utils import parse_quantile_column

AnomalyMethod = Literal["interval_breach", "conformal_interval", "mad_residual"]
AnomalyScore = Literal["margin", "normalized_margin", "zscore"]


@dataclass(frozen=True)
class AnomalyReport:
    """Anomaly detection report."""

    frame: pd.DataFrame
    method: str
    level: int
    score: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "level": self.level,
            "score": self.score,
            "summary": self.summary,
            "frame": self.frame.to_dict("records"),
        }


def detect_anomalies(
    forecast_with_y: pd.DataFrame,
    method: AnomalyMethod = "interval_breach",
    level: int = 99,
    score: AnomalyScore = "normalized_margin",
    calibrator: Any | None = None,
    strict: bool = False,
    id_col: str = "unique_id",
    ds_col: str = "ds",
    actual_col: str = "y",
    pred_col: str = "yhat",
) -> AnomalyReport:
    """Detect anomalies using forecast uncertainty."""
    if forecast_with_y.empty:
        raise EAnomalyFail("Forecast frame is empty.")

    if strict and calibrator is None:
        raise EAnomalyFail(
            "Strict mode requires calibrated intervals/quantiles for anomaly detection."
        )

    df = forecast_with_y.copy()

    lo_col = f"yhat_lo_{level}"
    hi_col = f"yhat_hi_{level}"
    if lo_col not in df.columns or hi_col not in df.columns:
        # Try to infer from quantiles
        lower_q = (1 - level / 100.0) / 2
        upper_q = 1 - lower_q
        q_cols = {parse_quantile_column(c): c for c in df.columns}
        lo_col = q_cols.get(lower_q)
        hi_col = q_cols.get(upper_q)
        if lo_col is None or hi_col is None:
            raise EAnomalyFail(
                "No interval/quantile columns available for anomaly detection.",
                context={"expected_level": level},
            )

    y = df[actual_col].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float) if pred_col in df.columns else None
    lo = df[lo_col].to_numpy(dtype=float)
    hi = df[hi_col].to_numpy(dtype=float)

    margin = np.zeros_like(y, dtype=float)
    margin = np.where(y < lo, lo - y, margin)
    margin = np.where(y > hi, y - hi, margin)

    if score == "margin":
        anomaly_score = margin
    elif score == "normalized_margin":
        width = np.maximum(hi - lo, 1e-8)
        anomaly_score = margin / width
    else:  # zscore
        if yhat is None:
            raise EAnomalyFail("yhat is required for zscore anomaly score.")
        width = np.maximum(hi - lo, 1e-8)
        anomaly_score = np.abs(y - yhat) / (0.5 * width)

    anomaly_flag = margin > 0

    frame = df[[id_col, ds_col, actual_col, pred_col]].copy()
    frame["lo"] = lo
    frame["hi"] = hi
    frame["anomaly"] = anomaly_flag
    frame["anomaly_score"] = anomaly_score
    frame["threshold"] = level
    frame["method"] = method
    frame["score"] = score

    summary = {
        "total": int(len(frame)),
        "anomalies": int(np.sum(anomaly_flag)),
        "anomaly_rate": float(np.mean(anomaly_flag)) if len(frame) > 0 else 0.0,
    }

    return AnomalyReport(
        frame=frame,
        method=method,
        level=level,
        score=score,
        summary=summary,
    )


__all__ = ["AnomalyReport", "detect_anomalies"]
