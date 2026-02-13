"""Stability monitoring for prediction jitter and quantile coverage."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from tsagentkit.monitoring.report import CalibrationReport, StabilityReport
from tsagentkit.utils import normalize_quantile_columns, quantile_col_name


class StabilityMonitor:
    """Monitor prediction stability and calibration.

    This class tracks:
    - Prediction jitter: Variance in point predictions across runs
    - Quantile coverage: Whether empirical coverage matches target quantiles

    Example:
        >>> monitor = StabilityMonitor(jitter_threshold=0.1)
        >>>
        >>> # Check jitter across multiple forecast runs
        >>> jitter = monitor.compute_jitter([forecast1, forecast2, forecast3])
        >>>
        >>> # Check quantile calibration
        >>> coverage = monitor.compute_coverage(actuals, forecasts, [0.1, 0.5, 0.9])
    """

    def __init__(
        self,
        jitter_threshold: float = 0.1,
        coverage_tolerance: float = 0.05,
    ):
        """Initialize stability monitor.

        Args:
            jitter_threshold: Coefficient of variation threshold for jitter warnings
            coverage_tolerance: Allowed deviation from target coverage
        """
        self.jitter_threshold = jitter_threshold
        self.coverage_tolerance = coverage_tolerance

    def compute_jitter(
        self,
        predictions: list[pd.DataFrame],
        method: Literal["cv", "mad"] = "cv",
    ) -> dict[str, float]:
        """Compute prediction jitter across multiple forecast runs.

        Jitter measures how much point predictions vary across different
        runs or model versions. High jitter indicates unstable predictions.

        Args:
            predictions: List of forecast DataFrames from different runs.
                        Each should have columns [unique_id, ds, yhat]
            method: "cv" for coefficient of variation, "mad" for median absolute deviation

        Returns:
            Dict mapping unique_id to jitter metric

        Example:
            >>> forecasts = [model.predict(data) for model in model_versions]
            >>> jitter = monitor.compute_jitter(forecasts, method="cv")
            >>> print(jitter["series_A"])
            0.05
        """
        if not predictions or len(predictions) < 2:
            return {}

        # Combine all predictions
        combined = predictions[0][["unique_id", "ds", "yhat"]].copy()
        combined.rename(columns={"yhat": "yhat_0"}, inplace=True)

        for i, pred in enumerate(predictions[1:], 1):
            combined = combined.merge(
                pred[["unique_id", "ds", "yhat"]].rename(columns={"yhat": f"yhat_{i}"}),
                on=["unique_id", "ds"],
                how="outer",
            )

        jitter_metrics = {}

        for uid in combined["unique_id"].unique():
            series_data = combined[combined["unique_id"] == uid]
            yhat_cols = [c for c in series_data.columns if c.startswith("yhat_")]

            if len(yhat_cols) < 2:
                continue

            values = series_data[yhat_cols].values

            if method == "cv":
                # Coefficient of variation (std / mean)
                means = np.nanmean(values, axis=1)
                stds = np.nanstd(values, axis=1)
                # Avoid division by zero
                cvs = np.where(
                    np.abs(means) > 1e-10,
                    stds / np.abs(means),
                    0.0
                )
                jitter = float(np.nanmean(cvs))
            else:  # mad
                # Median absolute deviation
                medians = np.nanmedian(values, axis=1, keepdims=True)
                mads = np.nanmedian(np.abs(values - medians), axis=1)
                jitter = float(np.nanmean(mads))

            jitter_metrics[uid] = jitter

        return jitter_metrics

    def compute_coverage(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        quantiles: list[float],
    ) -> dict[float, float]:
        """Compute empirical coverage for each quantile.

        Coverage is the proportion of actual values that fall below
        the predicted quantile. For a well-calibrated model:
        - q=0.1 should have ~10% coverage
        - q=0.5 should have ~50% coverage
        - q=0.9 should have ~90% coverage

        Args:
            actuals: DataFrame with actual values [unique_id, ds, y]
            forecasts: DataFrame with quantile forecasts [unique_id, ds, q_0.1, ...]
            quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])

        Returns:
            Dict mapping quantile to empirical coverage (0-1)

        Example:
            >>> coverage = monitor.compute_coverage(
            ...     actuals=df[["unique_id", "ds", "y"]],
            ...     forecasts=pred_df,
            ...     quantiles=[0.1, 0.5, 0.9]
            ... )
            >>> print(coverage[0.5])  # Should be ~0.5 for well-calibrated model
            0.52
        """
        forecasts = normalize_quantile_columns(forecasts)
        # Merge actuals with forecasts
        merged = actuals.merge(forecasts, on=["unique_id", "ds"], how="inner")

        coverage = {}
        for q in quantiles:
            col_name = quantile_col_name(q)
            if col_name not in merged.columns:
                continue

            # Compute coverage: proportion of actuals <= quantile prediction
            below_quantile = merged["y"] <= merged[col_name]
            coverage[q] = float(below_quantile.mean())

        return coverage

    def check_calibration(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        quantiles: list[float],
    ) -> CalibrationReport:
        """Check if quantiles are well-calibrated.

        Args:
            actuals: DataFrame with actual values [unique_id, ds, y]
            forecasts: DataFrame with quantile forecasts
            quantiles: List of target quantile levels

        Returns:
            CalibrationReport with coverage metrics and warnings
        """
        coverage = self.compute_coverage(actuals, forecasts, quantiles)

        # Compute calibration errors
        errors = {}
        for q in quantiles:
            if q in coverage:
                errors[q] = abs(coverage[q] - q)
            else:
                errors[q] = float("inf")

        # Check if all quantiles are well-calibrated
        well_calibrated = all(
            err <= self.coverage_tolerance
            for err in errors.values()
        )

        return CalibrationReport(
            target_quantiles=quantiles,
            empirical_coverage=coverage,
            calibration_errors=errors,
            well_calibrated=well_calibrated,
            tolerance=self.coverage_tolerance,
        )

    def generate_stability_report(
        self,
        predictions: list[pd.DataFrame] | None = None,
        actuals: pd.DataFrame | None = None,
        forecasts: pd.DataFrame | None = None,
        quantiles: list[float] | None = None,
    ) -> StabilityReport:
        """Generate a comprehensive stability report.

        Args:
            predictions: List of forecast DataFrames for jitter calculation
            actuals: DataFrame with actual values for coverage analysis
            forecasts: DataFrame with quantile forecasts for coverage analysis
            quantiles: List of quantile levels for coverage analysis

        Returns:
            StabilityReport with jitter and coverage metrics
        """
        # Compute jitter if predictions provided
        jitter_metrics = {}
        if predictions and len(predictions) >= 2:
            jitter_metrics = self.compute_jitter(predictions)

        overall_jitter = np.mean(list(jitter_metrics.values())) if jitter_metrics else 0.0

        # Identify high jitter series
        high_jitter_series = [
            uid for uid, jit in jitter_metrics.items()
            if jit > self.jitter_threshold
        ]

        # Compute coverage report if data provided
        coverage_report = None
        if actuals is not None and forecasts is not None and quantiles:
            coverage_report = self.check_calibration(actuals, forecasts, quantiles)

        return StabilityReport(
            jitter_metrics=jitter_metrics,
            overall_jitter=float(overall_jitter),
            jitter_threshold=self.jitter_threshold,
            high_jitter_series=high_jitter_series,
            coverage_report=coverage_report,
        )


def compute_prediction_interval_coverage(
    actuals: pd.Series,
    lower_bound: pd.Series,
    upper_bound: pd.Series,
) -> float:
    """Compute coverage for a prediction interval.

    Args:
        actuals: Actual values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval

    Returns:
        Proportion of actuals within the interval (0-1)

    Example:
        >>> coverage = compute_prediction_interval_coverage(
        ...     actuals=df["y"],
        ...     lower_bound=forecasts["q_0.1"],
        ...     upper_bound=forecasts["q_0.9"],
        ... )
        >>> print(coverage)  # Should be ~0.8 for 80% PI
        0.82
    """
    within_interval = (actuals >= lower_bound) & (actuals <= upper_bound)
    return float(within_interval.mean())
