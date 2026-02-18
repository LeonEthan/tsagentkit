"""Unified configuration for forecasting.

Replaces scattered parameters across TaskSpec, PanelContract, ForecastContract,
CovariateSpec, BacktestSpec, etc. with a single, minimal configuration class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ForecastConfig:
    """Minimal configuration for time-series forecasting.

    This unified config replaces the complex multi-spec approach with
    a single source of truth that uses sensible defaults.

    Args:
        h: Forecast horizon (number of periods to forecast)
        freq: Time-series frequency (pandas offset alias: 'D', 'H', 'M', etc.)
        quantiles: Quantile levels for probabilistic forecasts
        ensemble_method: How to aggregate ensemble forecasts ('median' or 'mean')
        min_tsfm: Minimum TSFMs required for ensemble
        fail_on_missing_tsfm: If True, abort if TSFM unavailable
        device: Device for TSFM inference ('auto', 'cuda', 'mps', 'cpu')
    """

    # Required
    h: int
    freq: str = "D"

    # Ensemble
    ensemble_method: Literal["median", "mean"] = "median"
    min_tsfm: int = 1  # Min TSFMs required

    # Output
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    # Hardware
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    def __post_init__(self) -> None:
        # Validation
        if self.h <= 0:
            raise ValueError(f"h must be positive, got {self.h}")
        if self.min_tsfm < 1:
            raise ValueError("min_tsfm must be at least 1")

    @staticmethod
    def quick(h: int, freq: str = "D") -> ForecastConfig:
        """Quick preset - minimal validation, allows fallback."""
        return ForecastConfig(h=h, freq=freq)

    @staticmethod
    def strict(h: int, freq: str = "D") -> ForecastConfig:
        """Strict preset - fail fast if TSFM unavailable."""
        return ForecastConfig(h=h, freq=freq, min_tsfm=1)

    @property
    def season_length(self) -> int | None:
        """Infer season length from frequency."""
        freq_map: dict[str, int] = {
            "D": 7,
            "B": 5,
            "H": 24,
            "h": 24,
            "T": 60,
            "min": 60,
            "M": 12,
            "ME": 12,
            "MS": 12,
            "Q": 4,
            "QS": 4,
            "W": 52,
        }
        base_freq = self.freq.lstrip("0123456789")
        return freq_map.get(base_freq)
