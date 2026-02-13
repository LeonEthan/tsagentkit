"""Calibration utilities for forecast intervals and quantiles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from tsagentkit.contracts import ECalibrationFail
from tsagentkit.utils import quantile_col_name, parse_quantile_column

CalibrationMethod = Literal["none", "conformal"]

CalibrationMethod = Literal["none", "conformal"]


def extract_quantiles(columns: Iterable[str]) -> list[float]:
    """Extract sorted quantile values from column names.

    Args:
        columns: Iterable of column names to parse

    Returns:
        Sorted list of unique quantile values found in column names

    Example:
        >>> extract_quantiles(["q0.1", "yhat", "q0.9", "q_95"])
        [0.1, 0.9, 0.95]
    """
    values = []
    for col in columns:
        q = parse_quantile_column(col)
        if q is not None:
            values.append(q)
    return sorted(set(values))


@dataclass(frozen=True)
class CalibratorArtifact:
    """Serializable calibration artifact.

    Attributes:
        method: Calibration method
        level: Target coverage level (percent)
        by: Calibration scope ("unique_id" or "global")
        deltas: Mapping from unique_id to calibration delta
        metadata: Additional metadata
    """

    method: CalibrationMethod
    level: int
    by: Literal["unique_id", "global"] = "unique_id"
    deltas: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_delta(self, unique_id: str | None = None) -> float:
        if self.by == "global":
            return float(self.deltas.get("global", 0.0))
        if unique_id is None:
            return float(self.deltas.get("global", 0.0))
        return float(self.deltas.get(unique_id, 0.0))


def fit_calibrator(
    cv: pd.DataFrame,
    method: CalibrationMethod = "conformal",
    level: int = 99,
    by: Literal["unique_id", "global"] = "unique_id",
    id_col: str = "unique_id",
    actual_col: str = "y",
    pred_col: str = "yhat",
) -> CalibratorArtifact:
    """Fit calibration artifact using out-of-sample residuals."""
    if method == "none":
        return CalibratorArtifact(method=method, level=level, by=by, deltas={"global": 0.0})

    if cv.empty:
        raise ECalibrationFail("CV frame is empty.", context={"method": method})

    if pred_col not in cv.columns or actual_col not in cv.columns:
        raise ECalibrationFail(
            "CV frame must include prediction and actual columns.",
            context={"pred_col": pred_col, "actual_col": actual_col},
        )

    residuals = np.abs(cv[actual_col].to_numpy(dtype=float) - cv[pred_col].to_numpy(dtype=float))
    if residuals.size == 0:
        raise ECalibrationFail("No residuals available for calibration.")

    alpha = level / 100.0
    deltas: dict[str, float] = {}

    if by == "global":
        deltas["global"] = float(np.quantile(residuals, alpha))
    else:
        for uid, group in cv.groupby(id_col):
            group_resid = np.abs(group[actual_col].to_numpy(dtype=float) - group[pred_col].to_numpy(dtype=float))
            if group_resid.size == 0:
                continue
            deltas[str(uid)] = float(np.quantile(group_resid, alpha))

        if not deltas:
            deltas["global"] = float(np.quantile(residuals, alpha))

    return CalibratorArtifact(
        method=method,
        level=level,
        by=by,
        deltas=deltas,
        metadata={
            "n_residuals": int(residuals.size),
            "alpha": alpha,
        },
    )


def apply_calibrator(
    forecast: pd.DataFrame,
    calibrator: CalibratorArtifact,
    id_col: str = "unique_id",
    pred_col: str = "yhat",
) -> pd.DataFrame:
    """Apply calibration artifact to a forecast frame."""
    if calibrator.method == "none":
        return forecast

    if forecast.empty:
        return forecast

    df = forecast.copy()
    level = calibrator.level
    lo_col = f"yhat_lo_{level}"
    hi_col = f"yhat_hi_{level}"

    if id_col not in df.columns or pred_col not in df.columns:
        raise ECalibrationFail(
            "Forecast frame must include unique_id and yhat for calibration.",
            context={"id_col": id_col, "pred_col": pred_col},
        )

    deltas = df[id_col].map(lambda uid: calibrator.get_delta(str(uid))).to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    df[lo_col] = yhat - deltas
    df[hi_col] = yhat + deltas

    # If quantile columns exist, adjust symmetrically around yhat.
    quantiles = extract_quantiles(df.columns)
    if quantiles:
        for q in quantiles:
            col = quantile_col_name(q)
            direction = -1.0 if q < 0.5 else 1.0
            df[col] = yhat + direction * deltas

    return df


__all__ = ["CalibratorArtifact", "fit_calibrator", "apply_calibrator"]
