"""Lightweight runtime telemetry for TSFM model loading.

Phase 0 baseline instrumentation:
- count model load calls
- accumulate model load time (ms)
- expose a process-local snapshot for benchmark reporting
"""

from __future__ import annotations

from threading import RLock
from typing import Any

_LOCK = RLock()
_STATE: dict[str, Any] = {
    "load_count": 0,
    "load_time_ms_total": 0.0,
    "per_adapter": {},
}


def reset_tsfm_runtime_stats() -> None:
    """Reset process-local TSFM runtime counters."""
    with _LOCK:
        _STATE["load_count"] = 0
        _STATE["load_time_ms_total"] = 0.0
        _STATE["per_adapter"] = {}


def record_tsfm_model_load(adapter_name: str, duration_ms: float) -> None:
    """Record one model load event."""
    with _LOCK:
        _STATE["load_count"] += 1
        _STATE["load_time_ms_total"] += float(duration_ms)

        per_adapter = _STATE["per_adapter"]
        if adapter_name not in per_adapter:
            per_adapter[adapter_name] = {
                "load_count": 0,
                "load_time_ms_total": 0.0,
            }
        per_adapter[adapter_name]["load_count"] += 1
        per_adapter[adapter_name]["load_time_ms_total"] += float(duration_ms)


def get_tsfm_runtime_stats() -> dict[str, Any]:
    """Return a snapshot of TSFM runtime counters."""
    with _LOCK:
        return {
            "load_count": int(_STATE["load_count"]),
            "load_time_ms_total": float(_STATE["load_time_ms_total"]),
            "per_adapter": {
                key: {
                    "load_count": int(value["load_count"]),
                    "load_time_ms_total": float(value["load_time_ms_total"]),
                }
                for key, value in _STATE["per_adapter"].items()
            },
        }

