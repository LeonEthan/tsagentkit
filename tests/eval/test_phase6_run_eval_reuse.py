"""Phase 6 tests for process-level predictor/session reuse in run_eval."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_RUN_EVAL_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "gift_eval" / "run_eval.py"
_SPEC = importlib.util.spec_from_file_location("gift_eval_run_eval_phase6", _RUN_EVAL_PATH)
assert _SPEC is not None and _SPEC.loader is not None

_fake_eval = types.ModuleType("eval")
_fake_eval.MED_LONG_DATASETS = []
_fake_eval.SHORT_DATASETS = []
_fake_eval.GIFTEvalRunner = object
_fake_eval.TSAgentKitPredictor = object
sys.modules["eval"] = _fake_eval

run_eval_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_eval_mod)


def test_main_reuses_one_predictor_across_multiple_datasets(monkeypatch, tmp_path) -> None:
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

    class _FakeRunner:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            _ = kwargs

        def evaluate(self, predictor, batch_size: int = 512, overwrite: bool = False):  # noqa: ANN001
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

    args = SimpleNamespace(
        dataset=None,
        datasets=None,
        term="short",
        all=True,
        mode="quick",
        storage_path=str(storage_path),
        output_path=str(output_path),
        batch_size=64,
        preload_adapters=["chronos", "moirai"],
        download=False,
        resume=False,
        max_datasets=None,
    )

    monkeypatch.setattr(run_eval_mod, "parse_args", lambda: args)
    monkeypatch.setattr(run_eval_mod, "resolve_configs", lambda _args: [("ds_a", "short"), ("ds_b", "short")])
    monkeypatch.setattr(run_eval_mod, "TSAgentKitPredictor", _FakePredictor)
    monkeypatch.setattr(run_eval_mod, "GIFTEvalRunner", _FakeRunner)
    monkeypatch.setattr(run_eval_mod, "reset_tsfm_runtime_stats", lambda: None)
    monkeypatch.setattr(
        run_eval_mod,
        "get_tsfm_runtime_stats",
        lambda: {"load_count": 0, "load_time_ms_total": 0.0, "per_adapter": {}},
    )

    exit_code = run_eval_mod.main()

    assert exit_code == 0
    assert _FakePredictor.init_calls == 1
    assert _FakePredictor.close_calls == 1
    assert len(predictor_ids) == 2
    assert predictor_ids[0] == predictor_ids[1]
