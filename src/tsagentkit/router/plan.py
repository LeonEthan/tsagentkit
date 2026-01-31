"""Execution plan for forecasting.

Defines the Plan dataclass which specifies the model execution strategy
including the fallback chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import hashlib
import json


@dataclass(frozen=True)
class Plan:
    """Execution plan for a forecasting task.

    Specifies the model selection strategy, fallback chain, and configuration
    for executing a forecasting pipeline.

    Attributes:
        primary_model: Name of the primary model to use
        fallback_chain: Ordered list of fallback models
        config: Model configuration dictionary
        strategy: Routing strategy used
        signature: Hash of the plan for provenance

    Examples:
        >>> plan = Plan(
        ...     primary_model="SeasonalNaive",
        ...     fallback_chain=["HistoricAverage"],
        ...     config={"season_length": 7},
        ... )
    """

    primary_model: str
    """Name of the primary model to attempt first."""

    fallback_chain: list[str] = field(default_factory=list)
    """Ordered list of fallback models to try on failure."""

    config: dict[str, Any] = field(default_factory=dict)
    """Model configuration parameters."""

    strategy: Literal["tsfm_first", "baseline_only", "auto"] = "auto"
    """Routing strategy that generated this plan."""

    signature: str = ""
    """Hash signature for provenance tracking."""

    def __post_init__(self) -> None:
        """Compute signature if not provided."""
        if not self.signature:
            # Use object.__setattr__ because dataclass is frozen
            sig = self._compute_signature()
            object.__setattr__(self, "signature", sig)

    def _compute_signature(self) -> str:
        """Compute hash signature of the plan.

        Returns:
            SHA-256 hash (truncated to 16 chars) of the plan contents
        """
        data = {
            "primary_model": self.primary_model,
            "fallback_chain": self.fallback_chain,
            "config": self.config,
            "strategy": self.strategy,
        }
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def get_all_models(self) -> list[str]:
        """Get all models in execution order.

        Returns:
            List of model names: primary + fallbacks
        """
        return [self.primary_model] + self.fallback_chain

    def to_signature(self) -> str:
        """Create human-readable signature.

        Returns:
            String like "Plan(SeasonalNaive->HistoricAverage)"
        """
        if self.fallback_chain:
            chain = "->".join([self.primary_model] + self.fallback_chain)
        else:
            chain = self.primary_model
        return f"Plan({chain})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_model": self.primary_model,
            "fallback_chain": self.fallback_chain,
            "config": self.config,
            "strategy": self.strategy,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """Create Plan from dictionary."""
        return cls(
            primary_model=data["primary_model"],
            fallback_chain=data.get("fallback_chain", []),
            config=data.get("config", {}),
            strategy=data.get("strategy", "auto"),
            signature=data.get("signature", ""),
        )
