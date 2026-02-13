"""Coverage monitoring for quantile forecasts.

Provides interval coverage checks to verify that prediction intervals
are well-calibrated over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CoverageCheck:
    """Interval coverage check for a specific quantile.

    Tracks the actual coverage rate compared to expected coverage
    for a given quantile level, with per-horizon breakdown.

    Attributes:
        quantile: The quantile level (e.g., 0.1 for lower bound)
        expected_coverage: Expected coverage rate
        actual_coverage: Actual observed coverage rate
        hit_rate_by_horizon: Coverage rate for each forecast horizon
        tolerance: Acceptable deviation from expected coverage
    """

    quantile: float
    expected_coverage: float
    actual_coverage: float
    hit_rate_by_horizon: dict[int, float]
    tolerance: float = 0.05

    def is_acceptable(self) -> bool:
        """Check if coverage is within tolerance of expected.

        For interval coverage, we want the actual coverage to be at least
        the expected coverage minus tolerance. Being slightly over is fine,
        but being under indicates the intervals are too narrow.
        """
        return self.actual_coverage >= (self.expected_coverage - self.tolerance)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "quantile": self.quantile,
            "expected_coverage": self.expected_coverage,
            "actual_coverage": self.actual_coverage,
            "hit_rate_by_horizon": self.hit_rate_by_horizon,
            "tolerance": self.tolerance,
            "is_acceptable": self.is_acceptable(),
        }


class CoverageMonitor:
    """Monitor quantile coverage over time.

    Provides functionality to check if prediction intervals are
    well-calibrated by comparing actual vs expected coverage rates.

    Example:
        >>> monitor = CoverageMonitor()
        >>> checks = monitor.check(
        ...     forecasts=forecast_df,
        ...     actuals=actual_df,
        ...     quantiles=[0.1, 0.5, 0.9],
        ... )
        >>> for check in checks:
        ...     print(f"Q{check.quantile}: {check.actual_coverage:.2%}")
    """

    def check(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        quantiles: list[float],
        tolerance: float = 0.05,
    ) -> list[CoverageCheck]:
        """Check coverage for given quantiles.

        Args:
            forecasts: Forecast dataframe with quantile columns (q_0.1, q_0.9, etc.)
            actuals: Actual values dataframe
            quantiles: List of quantile levels to check
            tolerance: Acceptable deviation from expected coverage

        Returns:
            List of CoverageCheck objects, one per quantile pair
        """
        results: list[CoverageCheck] = []

        # Ensure required columns exist
        if "unique_id" not in forecasts.columns or "ds" not in forecasts.columns:
            return results

        # Merge forecasts with actuals
        merged = forecasts.merge(
            actuals[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="inner",
        )

        if merged.empty:
            return results

        # Calculate coverage for each quantile pair
        for i, lower_q in enumerate(quantiles):
            for upper_q in quantiles[i + 1 :]:
                lower_col = f"q_{lower_q}"
                upper_col = f"q_{upper_q}"

                if lower_col not in merged.columns or upper_col not in merged.columns:
                    continue

                # Calculate overall coverage
                in_interval = (merged["y"] >= merged[lower_col]) & (
                    merged["y"] <= merged[upper_col]
                )
                actual_coverage = in_interval.mean()

                # Calculate coverage by horizon if available
                hit_rate_by_horizon: dict[int, float] = {}
                if "h" in merged.columns or "horizon" in merged.columns:
                    h_col = "h" if "h" in merged.columns else "horizon"
                    for h in sorted(merged[h_col].unique()):
                        h_data = merged[merged[h_col] == h]
                        if len(h_data) > 0:
                            h_in_interval = (h_data["y"] >= h_data[lower_col]) & (
                                h_data["y"] <= h_data[upper_col]
                            )
                            hit_rate_by_horizon[int(h)] = float(h_in_interval.mean())

                # Expected coverage is the difference between quantiles
                expected_coverage = upper_q - lower_q

                results.append(
                    CoverageCheck(
                        quantile=lower_q,
                        expected_coverage=expected_coverage,
                        actual_coverage=float(actual_coverage),
                        hit_rate_by_horizon=hit_rate_by_horizon,
                        tolerance=tolerance,
                    )
                )

        return results

    def check_single_quantile(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        quantile: float,
        tolerance: float = 0.05,
    ) -> CoverageCheck | None:
        """Check coverage for a single quantile (e.g., median).

        For a single quantile, checks if actuals fall below the quantile
        at the expected rate.

        Args:
            forecasts: Forecast dataframe
            actuals: Actual values dataframe
            quantile: Quantile level to check
            tolerance: Acceptable deviation

        Returns:
            CoverageCheck or None if data not available
        """
        q_col = f"q_{quantile}"
        if q_col not in forecasts.columns:
            return None

        merged = forecasts.merge(
            actuals[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="inner",
        )

        if merged.empty:
            return None

        # For single quantile: check if actual <= quantile at quantile rate
        below_quantile = merged["y"] <= merged[q_col]
        actual_coverage = float(below_quantile.mean())

        hit_rate_by_horizon: dict[int, float] = {}
        if "h" in merged.columns or "horizon" in merged.columns:
            h_col = "h" if "h" in merged.columns else "horizon"
            for h in sorted(merged[h_col].unique()):
                h_data = merged[merged[h_col] == h]
                if len(h_data) > 0:
                    h_below = h_data["y"] <= h_data[q_col]
                    hit_rate_by_horizon[int(h)] = float(h_below.mean())

        return CoverageCheck(
            quantile=quantile,
            expected_coverage=quantile,
            actual_coverage=actual_coverage,
            hit_rate_by_horizon=hit_rate_by_horizon,
            tolerance=tolerance,
        )
