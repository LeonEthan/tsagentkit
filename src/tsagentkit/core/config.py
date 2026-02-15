"""Unified configuration for forecasting.

Replaces scattered parameters across TaskSpec, PanelContract, ForecastContract,
CovariateSpec, BacktestSpec, etc. with a single, minimal configuration class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ForecastConfig:
    """Minimal configuration for time-series forecasting.

    This unified config replaces the complex multi-spec approach with
    a single source of truth that uses sensible defaults.

    Args:
        h: Forecast horizon (number of periods to forecast)
        freq: Time-series frequency (pandas offset alias: 'D', 'H', 'M', etc.)
        quantiles: Quantile levels for probabilistic forecasts
        id_col: Column name for series identifier
        time_col: Column name for timestamp
        target_col: Column name for target variable
        mode: Execution mode - affects backtest and strictness
        tsfm_mode: TSFM policy - 'required', 'preferred', or 'disabled'
        n_backtest_windows: Number of rolling windows for backtest (0 to skip)
        min_train_size: Minimum observations required per series
        allow_fallback: Whether to allow fallback to simpler models on failure
        ensemble_method: How to aggregate ensemble forecasts ('median' or 'mean')
        require_all_tsfm: If True, fail if any TSFM model fails
        min_models_for_ensemble: Minimum successful models required
    """

    # Core forecasting parameters
    h: int
    freq: str = "D"

    # Output configuration
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Column names (panel contract simplified)
    id_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"

    # Execution mode
    mode: Literal["quick", "standard", "strict"] = "standard"

    # TSFM policy (simplified from TSFMPolicy)
    tsfm_mode: Literal["required", "preferred", "disabled"] = "preferred"

    # Backtest configuration (simplified from BacktestSpec)
    n_backtest_windows: int = 5
    min_train_size: int = 56

    # Fallback behavior
    allow_fallback: bool = True

    # Ensemble configuration
    ensemble_method: Literal["median", "mean"] = "median"
    require_all_tsfm: bool = False
    min_models_for_ensemble: int = 1

    def __post_init__(self) -> None:
        # Validation
        if self.h <= 0:
            raise ValueError(f"h must be positive, got {self.h}")
        if self.n_backtest_windows < 0:
            raise ValueError(f"n_backtest_windows must be non-negative")
        if self.min_models_for_ensemble < 1:
            raise ValueError(f"min_models_for_ensemble must be at least 1")

    @classmethod
    def quick(cls, h: int, freq: str = "D") -> ForecastConfig:
        """Quick experimentation preset.

        Uses minimal backtest (2 windows), allows TSFM fallback to baselines.
        """
        return cls(
            h=h,
            freq=freq,
            mode="quick",
            tsfm_mode="preferred",
            n_backtest_windows=2,
            allow_fallback=True,
        )

    @classmethod
    def standard(cls, h: int, freq: str = "D") -> ForecastConfig:
        """Standard preset - balanced for most use cases."""
        return cls(
            h=h,
            freq=freq,
            mode="standard",
            tsfm_mode="required",
            n_backtest_windows=5,
            allow_fallback=True,
        )

    @classmethod
    def strict(cls, h: int, freq: str = "D") -> ForecastConfig:
        """Strict preset - fails fast on any issues.

        No auto-repair, no fallback, requires TSFM, all TSFM must succeed.
        """
        return cls(
            h=h,
            freq=freq,
            mode="strict",
            tsfm_mode="required",
            n_backtest_windows=5,
            allow_fallback=False,
            require_all_tsfm=True,
        )

    def with_covariates(
        self,
        static: list[str] | None = None,
        past: list[str] | None = None,
        future: list[str] | None = None,
    ) -> ForecastConfig:
        """Return config with covariate columns specified.

        Note: This is a simplified approach - covariates are referenced
        by column names in the main DataFrame or provided separately.
        """
        # Store covariate config in extra (for now)
        extra = getattr(self, "_extra", {})
        extra["covariates"] = {
            "static": static or [],
            "past": past or [],
            "future": future or [],
        }
        object.__setattr__(self, "_extra", extra)
        return self

    @property
    def season_length(self) -> int | None:
        """Infer season length from frequency."""
        freq_map: dict[str, int] = {
            "D": 7,
            "B": 5,
            "H": 24,
            "T": 60,
            "min": 60,
            "M": 12,
            "MS": 12,
            "Q": 4,
            "QS": 4,
            "W": 52,
        }
        base_freq = self.freq.lstrip("0123456789")
        return freq_map.get(base_freq)
