"""Tests for WS-4.4: quickstart module (forecast + diagnose) and WS-4.3: TaskSpec factories."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from tsagentkit.contracts.results import DryRunResult
from tsagentkit.contracts.task_spec import TaskSpec
from tsagentkit.quickstart import _prepare, diagnose


def _make_panel(
    n_series: int = 2,
    n_obs: int = 60,
    freq: str = "D",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a well-formed panel DataFrame."""
    np.random.seed(seed)
    rows = []
    for i in range(n_series):
        dates = pd.date_range("2023-01-01", periods=n_obs, freq=freq)
        for d in dates:
            rows.append({"unique_id": f"s{i}", "ds": d, "y": float(np.random.rand())})
    return pd.DataFrame(rows)


# ---- WS-4.3: TaskSpec factory methods ----------------------------------------


class TestTaskSpecFactories:
    """TaskSpec.starter() and TaskSpec.production() presets."""

    def test_starter_creates_valid_spec(self) -> None:
        spec = TaskSpec.starter(h=7)
        assert spec.h == 7
        assert spec.freq == "D"
        assert spec.tsfm_policy.mode == "preferred"
        assert spec.backtest.n_windows == 2

    def test_starter_custom_freq(self) -> None:
        spec = TaskSpec.starter(h=12, freq="H")
        assert spec.freq == "H"
        assert spec.h == 12

    def test_production_creates_valid_spec(self) -> None:
        spec = TaskSpec.production(h=14)
        assert spec.h == 14
        assert spec.freq == "D"
        assert spec.tsfm_policy.mode == "required"
        assert spec.backtest.n_windows == 5  # library default

    def test_production_custom_freq(self) -> None:
        spec = TaskSpec.production(h=30, freq="H")
        assert spec.freq == "H"

    def test_starter_is_classmethod(self) -> None:
        assert isinstance(TaskSpec.__dict__["starter"], classmethod)

    def test_production_is_classmethod(self) -> None:
        assert isinstance(TaskSpec.__dict__["production"], classmethod)

    def test_starter_hashable(self) -> None:
        spec = TaskSpec.starter(h=7)
        h = spec.model_hash()
        assert isinstance(h, str)
        assert len(h) == 16


# ---- WS-4.4: quickstart._prepare auto-rename + sort -------------------------


class TestPrepare:
    """Internal _prepare helper for column aliasing."""

    def test_rename_date_to_ds(self) -> None:
        df = pd.DataFrame(
            {
                "unique_id": ["s0"],
                "date": ["2023-01-01"],
                "y": [1.0],
            }
        )
        result = _prepare(df)
        assert "ds" in result.columns
        assert "date" not in result.columns

    def test_rename_value_to_y(self) -> None:
        df = pd.DataFrame(
            {
                "unique_id": ["s0"],
                "ds": ["2023-01-01"],
                "value": [1.0],
            }
        )
        result = _prepare(df)
        assert "y" in result.columns
        assert "value" not in result.columns

    def test_rename_id_to_unique_id(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["s0"],
                "ds": ["2023-01-01"],
                "y": [1.0],
            }
        )
        result = _prepare(df)
        assert "unique_id" in result.columns

    def test_no_rename_when_canonical(self) -> None:
        df = _make_panel(n_series=1, n_obs=5)
        result = _prepare(df)
        assert set(result.columns) == {"unique_id", "ds", "y"}

    def test_ds_coerced_to_datetime(self) -> None:
        df = pd.DataFrame(
            {
                "unique_id": ["s0"],
                "ds": ["2023-01-01"],
                "y": [1.0],
            }
        )
        result = _prepare(df)
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])

    def test_sorted_output(self) -> None:
        df = _make_panel(n_series=2, n_obs=10)
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        result = _prepare(shuffled)
        for uid in result["unique_id"].unique():
            subset = result[result["unique_id"] == uid]
            assert list(subset["ds"]) == sorted(subset["ds"])


# ---- WS-4.4: diagnose() (dry_run integration) ---------------------------


class TestDiagnose:
    """diagnose() returns a DryRunResult dict without fitting."""

    def test_diagnose_returns_dict(self) -> None:
        df = _make_panel(n_series=2, n_obs=60)
        result = diagnose(df, freq="D", horizon=7)
        assert isinstance(result, dict)

    def test_diagnose_keys(self) -> None:
        df = _make_panel(n_series=2, n_obs=60)
        result = diagnose(df, freq="D", horizon=7)
        expected_keys = {"validation", "qa_report", "plan", "route_decision", "task_spec_used"}
        assert expected_keys == set(result.keys())

    def test_diagnose_validation_valid(self) -> None:
        df = _make_panel(n_series=2, n_obs=60)
        result = diagnose(df, freq="D", horizon=7)
        assert result["validation"]["valid"] is True

    def test_diagnose_plan_has_candidates(self) -> None:
        df = _make_panel(n_series=2, n_obs=60)
        result = diagnose(df, freq="D", horizon=7)
        assert "candidate_models" in result["plan"]
        assert len(result["plan"]["candidate_models"]) > 0


# ---- WS-3.4: DryRunResult dataclass -----------------------------------------


class TestDryRunResult:
    """DryRunResult dataclass."""

    def test_creation(self) -> None:
        dr = DryRunResult(
            validation={"valid": True},
            qa_report={"issues": []},
            plan={"candidate_models": ["Naive"]},
            route_decision={"buckets": []},
            task_spec_used={"h": 7},
        )
        assert dr.validation == {"valid": True}

    def test_to_dict(self) -> None:
        dr = DryRunResult(
            validation={"valid": True},
            qa_report={"issues": []},
            plan={"candidate_models": ["Naive"]},
            route_decision={"buckets": []},
        )
        d = dr.to_dict()
        assert set(d.keys()) == {
            "validation",
            "qa_report",
            "plan",
            "route_decision",
            "task_spec_used",
        }

    def test_frozen(self) -> None:
        dr = DryRunResult(
            validation={"valid": True},
            qa_report={"issues": []},
            plan={},
            route_decision={},
        )
        with pytest.raises(FrozenInstanceError):
            dr.validation = {"valid": False}  # type: ignore[misc]
