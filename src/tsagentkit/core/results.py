"""Result types for forecasting.

Simplified output containers that preserve essential information
while minimizing complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """Forecast output with minimal structure.

    Replaces the complex ForecastResult + ForecastFrame + ModelArtifact
    hierarchy with a single, clear result type.
    """

    df: pd.DataFrame  # [unique_id, ds, yhat, ...quantiles]
    model_name: str
    config: Any | None = None  # ForecastConfig reference

    def get_series(self, unique_id: str) -> pd.DataFrame:
        """Get forecast for a single series."""
        return self.df[self.df["unique_id"] == unique_id].copy()

    @property
    def point_forecast(self) -> pd.DataFrame:
        """Return point forecasts (yhat column)."""
        cols = ["unique_id", "ds", "yhat"]
        available = [c for c in cols if c in self.df.columns]
        return self.df[available].copy()


@dataclass
class RunResult:
    """Complete run result including forecast and metadata.

    Simplified from RunArtifact which had 20+ fields of varying importance.
    """

    forecast: ForecastResult
    duration_ms: float
    model_used: str

    # Optional components (None if not computed)
    backtest_metrics: dict[str, float] | None = None
    fallbacks: list[dict[str, str]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return forecast as DataFrame (convenience method)."""
        return self.forecast.df

    def summary(self) -> dict[str, Any]:
        """Human-readable summary of the run."""
        return {
            "model": self.model_used,
            "duration_ms": round(self.duration_ms, 2),
            "fallbacks": len(self.fallbacks),
            "forecast_shape": self.forecast.df.shape,
        }
