"""Task specification for forecasting tasks.

Defines the TaskSpec class which specifies all parameters for a forecasting run.
Must be JSON-serializable and hashable for provenance tracking.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskSpec(BaseModel):
    """Specification for a forecasting task.

    This class defines all parameters needed to execute a forecasting pipeline.
    It is designed to be JSON-serializable and hashable for provenance tracking.

    Attributes:
        horizon: Number of steps to forecast
        freq: Frequency of the time series (pandas freq string)
        rolling_step: Step size for rolling backtest windows
        quantiles: List of quantiles to forecast (optional)
        covariate_policy: How to handle covariates ('ignore', 'known', 'auto')
        repair_strategy: QA repair strategy configuration (optional)
        season_length: Seasonal period (auto-detected if None)
        valid_from: Start date for validation period (optional)
        valid_until: End date for validation period (optional)
        metadata: Additional user-defined metadata
        seed: Random seed for reproducibility (optional)

    Examples:
        >>> spec = TaskSpec(horizon=7, freq="D")
        >>> spec = TaskSpec(
        ...     horizon=24,
        ...     freq="H",
        ...     quantiles=[0.1, 0.5, 0.9],
        ...     covariate_policy="known"
        ... )
    """

    # Required fields
    horizon: int = Field(
        ...,
        ge=1,
        description="Number of steps to forecast",
    )
    freq: str = Field(
        ...,
        description="Frequency of the time series (pandas freq string, e.g., 'D', 'H', 'M')",
    )

    # Backtest configuration
    rolling_step: int | None = Field(
        default=None,
        ge=1,
        description="Step size for rolling backtest windows (defaults to horizon)",
    )

    # Forecast configuration
    quantiles: list[float] | None = Field(
        default=None,
        description="List of quantiles to forecast (e.g., [0.1, 0.5, 0.9])",
    )

    # Covariate handling
    covariate_policy: Literal["ignore", "known", "observed", "auto"] = Field(
        default="ignore",
        description="How to handle covariates: 'ignore' (don't use), "
                    "'known' (all covariates known in advance), "
                    "'observed' (covariates observed up to forecast time), "
                    "'auto' (infer from data)",
    )

    # QA repair strategy
    repair_strategy: dict[str, Any] | None = Field(
        default=None,
        description="Optional QA repair configuration (e.g., interpolate_missing, "
                    "winsorize_outliers, missing_method)",
    )

    # Seasonality
    season_length: int | None = Field(
        default=None,
        ge=1,
        description="Seasonal period (auto-detected from freq if None)",
    )

    # Validation period (optional)
    valid_from: str | None = Field(
        default=None,
        description="Start date for validation period (ISO 8601 format)",
    )
    valid_until: str | None = Field(
        default=None,
        description="End date for validation period (ISO 8601 format)",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional user-defined metadata (must be JSON-serializable)",
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (applies to stochastic operations)",
    )

    @model_validator(mode="before")
    @classmethod
    def _set_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Set default values derived from other fields before validation."""
        # Make a copy to avoid modifying the original
        data = dict(data)

        horizon = data.get("horizon")
        freq = data.get("freq")

        # Set rolling_step default
        if data.get("rolling_step") is None and horizon is not None:
            data["rolling_step"] = horizon

        # Infer season_length
        if data.get("season_length") is None and freq is not None:
            data["season_length"] = cls._infer_season_length(freq)

        # Process quantiles
        quantiles = data.get("quantiles")
        if quantiles is not None:
            # Ensure quantiles are sorted and unique
            sorted_unique = sorted(set(quantiles))
            data["quantiles"] = sorted_unique
            # Validate quantile range
            for q in sorted_unique:
                if not 0 < q < 1:
                    raise ValueError(f"Quantile {q} must be between 0 and 1")

        return data

    @model_validator(mode="after")
    def _validate_dates(self) -> TaskSpec:
        """Validate that valid_from < valid_until if both provided."""
        if self.valid_from and self.valid_until and self.valid_from >= self.valid_until:
            raise ValueError("valid_from must be before valid_until")
        return self

    @staticmethod
    def _infer_season_length(freq: str) -> int | None:
        """Infer season length from frequency string.

        Args:
            freq: Pandas frequency string

        Returns:
            Inferred season length or None if cannot infer
        """
        # Common frequency to season length mappings
        freq_map: dict[str, int] = {
            "D": 7,      # Daily -> weekly
            "B": 5,      # Business days -> weekly
            "H": 24,     # Hourly -> daily
            "T": 60,     # Minutely -> hourly (simplified)
            "min": 60,
            "M": 12,     # Monthly -> yearly
            "MS": 12,
            "Q": 4,      # Quarterly -> yearly
            "QS": 4,
            "W": 52,     # Weekly -> yearly (approximate)
        }

        # Extract base frequency (remove numbers prefix)
        base_freq = freq.lstrip("0123456789")
        return freq_map.get(base_freq)

    def model_hash(self) -> str:
        """Compute a hash of the task spec for provenance.

        Returns:
            SHA-256 hash of the JSON representation
        """
        import hashlib
        import json

        # Get JSON-serializable dict (excluding metadata for core hash)
        data = self.model_dump(exclude={"metadata"}, exclude_none=True)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_signature(self) -> str:
        """Create a human-readable signature for provenance.

        Returns:
            String signature like "TaskSpec(H=7,f=D,q=[0.1,0.5,0.9])"
        """
        parts = [f"H={self.horizon}", f"f={self.freq}"]
        if self.quantiles:
            q_str = ",".join(f"{q:.2f}" for q in self.quantiles)
            parts.append(f"q=[{q_str}]")
        if self.season_length:
            parts.append(f"s={self.season_length}")
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return f"TaskSpec({','.join(parts)})"

    def set_random_seed(self) -> None:
        """Set random seed for reproducibility.

        This method sets the random seed for numpy, random, and other
        stochastic libraries to ensure reproducible results.

        Example:
            >>> spec = TaskSpec(horizon=7, freq="D", seed=42)
            >>> spec.set_random_seed()  # Sets all random seeds
        """
        if self.seed is None:
            return

        import random

        import numpy as np

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Try to set torch seed if available
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

    model_config = ConfigDict(
        frozen=True,  # Makes the model hashable and immutable
        extra="forbid",  # Prevent extra fields
    )
