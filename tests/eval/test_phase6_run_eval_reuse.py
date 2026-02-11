"""Phase 6 tests for process-level predictor/session reuse in gift-eval orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsagentkit.gift_eval import eval as gift_eval_mod


def test_run_combinations_reuses_one_predictor_across_multiple_datasets(monkeypatch, tmp_path) -> None:
    storage_path = tmp_path / "data"
    output_path = tmp_path / "results"
    storage_path.mkdir(parents=True)

    class _FakePredictor:
        init_calls = 0
        close_calls = 0

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            _ = kwargs
            type(self).init_calls += 1

        def close(self) -> None:
            type(self).close_calls += 1

    predictor_ids: list[int] = []

    class _FakeEvaluator:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            _ = kwargs

        def evaluate_predictor(self, predictor, batch_size: int = 512, overwrite: bool = False):  # noqa: ANN001
            _ = batch_size
            _ = overwrite
            predictor_ids.append(id(predictor))
            return pd.DataFrame(
                {
                    "eval_metrics/MASE[0.5]": [1.0],
                    "eval_metrics/sMAPE[0.5]": [2.0],
                    "eval_metrics/mean_weighted_sum_quantile_loss": [3.0],
                }
            )

    monkeypatch.setattr(gift_eval_mod, "TSAgentKitPredictor", _FakePredictor)
    monkeypatch.setattr(gift_eval_mod, "GIFTEval", _FakeEvaluator)

    result_path = gift_eval_mod.run_combinations(
        combinations=[("ds_a", "short"), ("ds_b", "short")],
        storage_path=storage_path,
        output_path=output_path,
        mode="quick",
        batch_size=64,
        preload_adapters=["chronos", "moirai"],
    )

    assert result_path == Path(output_path) / "all_results.csv"
    assert _FakePredictor.init_calls == 1
    assert _FakePredictor.close_calls == 1
    assert len(predictor_ids) == 2
    assert predictor_ids[0] == predictor_ids[1]
