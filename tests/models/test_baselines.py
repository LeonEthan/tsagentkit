from __future__ import annotations

from tsagentkit.models.baselines import is_baseline_model


def test_baseline_model_aliases() -> None:
    assert is_baseline_model("ETS")
    assert is_baseline_model("AutoETS")
    assert is_baseline_model("MovingAverage")
    assert is_baseline_model("WindowAverage")
