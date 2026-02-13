"""Tests for WS-3.3: standalone repair() function."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsagentkit.contracts.task_spec import PanelContract, TaskSpec
from tsagentkit.repair import repair


def _make_panel(
    n_series: int = 2,
    n_obs: int = 30,
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


class TestRepairSort:
    """Sorting repairs."""

    def test_unsorted_gets_sorted(self) -> None:
        df = _make_panel()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        repaired, actions = repair(shuffled)
        action_types = [a["action"] for a in actions]
        assert "sort" in action_types
        # Verify actually sorted
        for uid in repaired["unique_id"].unique():
            subset = repaired[repaired["unique_id"] == uid]
            assert list(subset["ds"]) == sorted(subset["ds"])

    def test_already_sorted_no_sort_action(self) -> None:
        df = _make_panel()
        _, actions = repair(df)
        action_types = [a["action"] for a in actions]
        assert "sort" not in action_types


class TestRepairDuplicates:
    """Duplicate removal repairs."""

    def test_duplicates_removed(self) -> None:
        df = _make_panel()
        dup = pd.concat([df, df.iloc[:3]], ignore_index=True)
        repaired, actions = repair(dup)
        action_types = [a["action"] for a in actions]
        assert "drop_duplicates" in action_types
        # Should have no duplicates
        assert not repaired.duplicated(subset=["unique_id", "ds"]).any()

    def test_no_duplicates_no_action(self) -> None:
        df = _make_panel()
        _, actions = repair(df)
        action_types = [a["action"] for a in actions]
        assert "drop_duplicates" not in action_types


class TestRepairFutureRows:
    """Future row dropping."""

    def test_future_null_rows_dropped(self) -> None:
        df = _make_panel(n_series=1, n_obs=10)
        # Add future rows with null y
        future = pd.DataFrame(
            {
                "unique_id": ["s0"] * 3,
                "ds": pd.date_range("2023-01-11", periods=3, freq="D"),
                "y": [np.nan, np.nan, np.nan],
            }
        )
        combined = pd.concat([df, future], ignore_index=True)
        repaired, actions = repair(combined)
        action_types = [a["action"] for a in actions]
        assert "drop_future_rows" in action_types
        assert len(repaired) == 10


class TestRepairWithSpec:
    """Using TaskSpec to resolve panel columns."""

    def test_with_task_spec(self) -> None:
        df = _make_panel()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        spec = TaskSpec(h=7, freq="D")
        repaired, actions = repair(shuffled, spec=spec)
        assert len(repaired) == len(df)

    def test_with_panel_contract(self) -> None:
        df = _make_panel()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        repaired, actions = repair(shuffled, panel_contract=PanelContract())
        assert len(repaired) == len(df)


class TestRepairEmpty:
    """Edge case: empty DataFrame."""

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["unique_id", "ds", "y"])
        repaired, actions = repair(df)
        assert len(repaired) == 0


class TestRepairReturnStructure:
    """Return value structure."""

    def test_returns_tuple(self) -> None:
        df = _make_panel()
        result = repair(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_actions_are_dicts(self) -> None:
        df = _make_panel()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        _, actions = repair(shuffled)
        for action in actions:
            assert isinstance(action, dict)
            assert "action" in action
            assert "description" in action
            assert "details" in action

    def test_original_not_mutated(self) -> None:
        df = _make_panel()
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        original = shuffled.copy()
        repair(shuffled)
        pd.testing.assert_frame_equal(shuffled, original)
