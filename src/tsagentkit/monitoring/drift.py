"""Drift detection using PSI and KS tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tsagentkit.monitoring.report import DriftReport, FeatureDriftResult

if TYPE_CHECKING:
    pass


class DriftDetector:
    """Detect data drift between reference and current distributions.

    Supports two methods:
    - PSI (Population Stability Index): Industry standard for distribution drift
    - KS (Kolmogorov-Smirnov): Statistical test for distribution differences

    PSI interpretation:
    - < 0.1: No significant change
    - 0.1 - 0.2: Moderate change
    - > 0.2: Significant change (drift detected)

    KS interpretation:
    - p-value < 0.05: Statistically significant difference (drift detected)

    Example:
        >>> detector = DriftDetector(method="psi", threshold=0.2)
        >>> report = detector.detect(reference_data, current_data, features=["sales"])
        >>> if report.drift_detected:
        ...     print(f"Drift detected in features: {report.get_drifting_features()}")
    """

    def __init__(
        self,
        method: Literal["psi", "ks"] = "psi",
        threshold: float | None = None,
        n_bins: int = 10,
    ):
        """Initialize drift detector.

        Args:
            method: Drift detection method ("psi" or "ks")
            threshold: Threshold for drift detection.
                      Default: 0.2 for PSI, 0.05 for KS (p-value)
            n_bins: Number of bins for PSI calculation

        Raises:
            ValueError: If invalid method specified
        """
        if method not in ("psi", "ks"):
            raise ValueError(f"Method must be 'psi' or 'ks', got {method}")

        self.method = method
        self.n_bins = n_bins

        # Set default thresholds
        if threshold is None:
            self.threshold = 0.2 if method == "psi" else 0.05
        else:
            self.threshold = threshold

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> DriftReport:
        """Detect drift between reference and current datasets.

        Args:
            reference_data: Baseline/reference distribution (training data)
            current_data: Current data to compare (recent observations)
            features: List of features to check (defaults to numeric columns)

        Returns:
            DriftReport with per-feature and overall results

        Example:
            >>> detector = DriftDetector(method="psi")
            >>> report = detector.detect(
            ...     reference_data=train_df,
            ...     current_data=recent_df,
            ...     features=["sales", "price"]
            ... )
            >>> print(report.overall_drift_score)
            0.15
        """
        # Auto-select numeric features if not specified
        if features is None:
            features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Exclude common non-feature columns
            exclude = {"unique_id", "ds", "y", "timestamp"}
            features = [f for f in features if f not in exclude]

        feature_drifts: dict[str, FeatureDriftResult] = {}
        drift_scores = []

        for feature in features:
            if feature not in reference_data.columns:
                continue
            if feature not in current_data.columns:
                continue

            result = self._analyze_feature(
                reference_data[feature],
                current_data[feature],
                feature,
            )
            feature_drifts[feature] = result
            drift_scores.append(result.statistic)

        # Calculate overall drift score (mean of individual statistics)
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0

        # Determine if drift detected based on method
        if self.method == "psi":
            drift_detected = overall_drift > self.threshold
        else:  # ks
            # For KS, drift detected if any feature has p-value < threshold
            # Overall score is the max KS statistic across features
            p_values = [
                r.p_value for r in feature_drifts.values()
                if r.p_value is not None
            ]
            if p_values:
                # Drift detected if any p-value is below threshold
                drift_detected = any(p < self.threshold for p in p_values)
            else:
                drift_detected = False

        return DriftReport(
            drift_detected=drift_detected,
            feature_drifts=feature_drifts,
            overall_drift_score=float(overall_drift),
            threshold_used=self.threshold,
        )

    def _analyze_feature(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Analyze drift for a single feature.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature

        Returns:
            FeatureDriftResult with drift statistics
        """
        # Remove NaN values
        ref_values = reference.dropna().values
        cur_values = current.dropna().values

        if self.method == "psi":
            statistic = self._compute_psi(ref_values, cur_values, self.n_bins)
            p_value = None
            drift_detected = statistic > self.threshold
        else:  # ks
            statistic, p_value = self._compute_ks_test(ref_values, cur_values)
            # For KS, drift detected if p-value < threshold
            drift_detected = p_value < self.threshold

        # Compute distribution summaries
        ref_dist = self._summarize_distribution(ref_values)
        cur_dist = self._summarize_distribution(cur_values)

        return FeatureDriftResult(
            feature_name=feature_name,
            metric=self.method,
            statistic=float(statistic),
            p_value=float(p_value) if p_value is not None else None,
            drift_detected=drift_detected,
            reference_distribution=ref_dist,
            current_distribution=cur_dist,
        )

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index.

        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))

        PSI interpretation:
        - < 0.1: No significant change
        - 0.1 - 0.2: Moderate change
        - > 0.2: Significant change

        Args:
            reference: Reference distribution values
            current: Current distribution values
            n_bins: Number of bins for discretization

        Returns:
            PSI value (float)
        """
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create bins based on reference distribution
        min_val, max_val = reference.min(), reference.max()

        # Handle constant reference
        if min_val == max_val:
            return 0.0 if current.min() == current.max() == min_val else 1.0

        # Create bins
        bins = np.linspace(min_val, max_val, n_bins + 1)
        bins[-1] += 1e-10  # Ensure max value is included

        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Convert to probabilities
        ref_pct = ref_hist / len(reference) + 1e-10  # Add epsilon to avoid log(0)
        cur_pct = cur_hist / len(current) + 1e-10

        # Compute PSI
        psi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        psi = np.sum(psi_values)

        return float(psi)

    def _compute_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """Compute Kolmogorov-Smirnov test.

        Args:
            reference: Reference distribution values
            current: Current distribution values

        Returns:
            Tuple of (statistic, p_value)
        """
        from scipy import stats

        if len(reference) == 0 or len(current) == 0:
            return 0.0, 1.0

        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    def _summarize_distribution(self, values: np.ndarray) -> dict:
        """Create a summary of a distribution.

        Args:
            values: Array of values

        Returns:
            Dictionary with distribution statistics
        """
        if len(values) == 0:
            return {"count": 0}

        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }


def compute_psi_summary(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute detailed PSI breakdown by bin.

    Args:
        reference: Reference distribution
        current: Current distribution
        n_bins: Number of bins

    Returns:
        Dictionary with PSI breakdown
    """
    detector = DriftDetector(method="psi", n_bins=n_bins)

    ref_values = np.asarray(reference.dropna() if hasattr(reference, "dropna") else reference)
    cur_values = np.asarray(current.dropna() if hasattr(current, "dropna") else current)

    if len(ref_values) == 0 or len(cur_values) == 0:
        return {"psi": 0.0, "bins": []}

    # Create bins
    min_val, max_val = ref_values.min(), ref_values.max()
    if min_val == max_val:
        return {"psi": 0.0, "bins": [], "message": "Constant reference distribution"}

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[-1] += 1e-10

    # Compute histograms
    ref_hist, bin_edges = np.histogram(ref_values, bins=bins)
    cur_hist, _ = np.histogram(cur_values, bins=bins)

    ref_pct = ref_hist / len(ref_values) + 1e-10
    cur_pct = cur_hist / len(cur_values) + 1e-10

    # Compute per-bin PSI
    bin_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    bins_data = []
    for i in range(n_bins):
        bins_data.append({
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "reference_count": int(ref_hist[i]),
            "reference_pct": float(ref_pct[i] - 1e-10),
            "current_count": int(cur_hist[i]),
            "current_pct": float(cur_pct[i] - 1e-10),
            "psi": float(bin_psi[i]),
        })

    return {
        "psi": float(np.sum(bin_psi)),
        "bins": bins_data,
    }
