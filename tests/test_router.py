"""Tests for router and plan functionality.

Tests plan building, model candidate selection, and plan execution.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsagentkit import ForecastConfig, TSDataset
from tsagentkit.core.errors import EModelFailed, ETSFMRequired
from tsagentkit.router import (
    ModelCandidate,
    Plan,
    build_plan,
    inspect_tsfm_adapters,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    return pd.DataFrame({
        "unique_id": ["A"] * 30,
        "ds": pd.date_range("2024-01-01", periods=30),
        "y": range(30),
    })


@pytest.fixture
def config():
    """Create default config."""
    return ForecastConfig(h=7, freq="D")


@pytest.fixture
def dataset(sample_df, config):
    """Create TSDataset."""
    return TSDataset.from_dataframe(sample_df, config)


class TestModelCandidate:
    """Test ModelCandidate dataclass."""

    def test_create_statistical_candidate(self):
        """Create statistical model candidate."""
        candidate = ModelCandidate(name="SeasonalNaive")
        assert candidate.name == "SeasonalNaive"
        assert candidate.is_tsfm is False
        assert candidate.adapter_name is None
        assert candidate.params == {}

    def test_create_tsfm_candidate(self):
        """Create TSFM model candidate."""
        candidate = ModelCandidate(name="tsfm-chronos", is_tsfm=True, adapter_name="chronos")
        assert candidate.name == "tsfm-chronos"
        assert candidate.is_tsfm is True
        assert candidate.adapter_name == "chronos"

    def test_create_with_params(self):
        """Create candidate with parameters."""
        candidate = ModelCandidate(
            name="CustomModel",
            params={"learning_rate": 0.01, "epochs": 100}
        )
        assert candidate.params == {"learning_rate": 0.01, "epochs": 100}


class TestPlan:
    """Test Plan dataclass."""

    def test_create_empty_plan(self):
        """Create empty plan."""
        plan = Plan()
        assert plan.tsfm_models == []
        assert plan.statistical_models == []
        assert plan.ensemble_method == "median"
        assert plan.min_models_for_ensemble == 1

    def test_create_plan_with_models(self):
        """Create plan with models."""
        tsfm_models = [ModelCandidate(name="tsfm-chronos", is_tsfm=True)]
        stat_models = [ModelCandidate(name="Naive")]
        plan = Plan(tsfm_models=tsfm_models, statistical_models=stat_models)

        assert len(plan.tsfm_models) == 1
        assert len(plan.statistical_models) == 1

    def test_all_models(self):
        """all_models returns all models."""
        tsfm_models = [
            ModelCandidate(name="tsfm-chronos", is_tsfm=True),
            ModelCandidate(name="tsfm-moirai", is_tsfm=True),
        ]
        stat_models = [
            ModelCandidate(name="Naive"),
            ModelCandidate(name="SeasonalNaive"),
        ]
        plan = Plan(tsfm_models=tsfm_models, statistical_models=stat_models)

        all_models = plan.all_models()
        assert len(all_models) == 4

    def test_plan_is_frozen(self):
        """Plan dataclass is frozen."""
        plan = Plan()
        with pytest.raises(AttributeError):
            plan.ensemble_method = "mean"


class TestInspectTSFMAdapters:
    """Test inspect_tsfm_adapters function."""

    def test_returns_list(self):
        """Function returns a list."""
        adapters = inspect_tsfm_adapters()
        assert isinstance(adapters, list)

    def test_adapters_are_strings(self):
        """All adapters are strings."""
        adapters = inspect_tsfm_adapters()
        for adapter in adapters:
            assert isinstance(adapter, str)

    def test_known_adapters_exist(self):
        """Known adapters exist in codebase."""
        adapters = inspect_tsfm_adapters()
        expected = ["chronos", "timesfm", "moirai"]
        for expected_adapter in expected:
            assert expected_adapter in adapters


class TestBuildPlan:
    """Test build_plan function."""

    def test_build_plan_disabled_tsfm(self, dataset):
        """Build plan with TSFM disabled."""
        plan = build_plan(dataset, tsfm_mode="disabled", allow_fallback=True)

        assert len(plan.tsfm_models) == 0
        assert len(plan.statistical_models) == 3  # SeasonalNaive, HistoricAverage, Naive

    def test_build_plan_no_fallback(self, dataset):
        """Build plan without fallback models."""
        plan = build_plan(dataset, tsfm_mode="disabled", allow_fallback=False)

        assert len(plan.tsfm_models) == 0
        assert len(plan.statistical_models) == 0

    def test_build_plan_ensemble_config(self, dataset):
        """Build plan with ensemble configuration."""
        plan = build_plan(
            dataset,
            tsfm_mode="disabled",
            ensemble_method="mean",
            require_all_tsfm=True,
        )

        assert plan.ensemble_method == "mean"
        assert plan.require_all_tsfm is True

    def test_build_plan_tsfm_required_raises(self, dataset):
        """TSFM required but none available raises ETSFMRequired."""
        # Mock no TSFM adapters available
        import tsagentkit.router.plan as plan_module
        original_inspect = plan_module.inspect_tsfm_adapters

        def mock_inspect():
            return []

        plan_module.inspect_tsfm_adapters = mock_inspect

        try:
            with pytest.raises(ETSFMRequired) as exc_info:
                build_plan(dataset, tsfm_mode="required")
            assert "TSFM required but no adapters available" in str(exc_info.value)
        finally:
            plan_module.inspect_tsfm_adapters = original_inspect

    def test_build_plan_tsfm_preferred_no_adapters(self, dataset):
        """TSFM preferred with no adapters falls back to statistical."""
        import tsagentkit.router.plan as plan_module
        original_inspect = plan_module.inspect_tsfm_adapters

        def mock_inspect():
            return []

        plan_module.inspect_tsfm_adapters = mock_inspect

        try:
            plan = build_plan(dataset, tsfm_mode="preferred", allow_fallback=True)
            assert len(plan.tsfm_models) == 0
            assert len(plan.statistical_models) == 3
        finally:
            plan_module.inspect_tsfm_adapters = original_inspect

    def test_build_plan_with_available_tsfm(self, dataset):
        """Build plan includes available TSFM adapters."""
        plan = build_plan(dataset, tsfm_mode="preferred", allow_fallback=True)

        # Should have TSFM models (chronos, timesfm, moirai)
        assert len(plan.tsfm_models) >= 3
        # Should also have statistical models
        assert len(plan.statistical_models) == 3

    def test_build_plan_min_models_for_ensemble(self, dataset):
        """Plan has min_models_for_ensemble set."""
        plan = build_plan(dataset, tsfm_mode="disabled", allow_fallback=True)
        assert plan.min_models_for_ensemble == 1


class TestPlanExecution:
    """Test Plan.execute method."""

    def test_execute_empty_plan_raises(self, sample_df, config):
        """Execute empty plan raises EModelFailed."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        plan = Plan(tsfm_models=[], statistical_models=[])

        with pytest.raises(EModelFailed, match="Insufficient models"):
            plan.execute(dataset)

    def test_execute_statistical_models(self, sample_df, config):
        """Execute plan with statistical models."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        plan = build_plan(dataset, tsfm_mode="disabled", allow_fallback=True)

        result = plan.execute(dataset)

        assert "artifacts" in result
        assert len(result["artifacts"]) == 3  # 3 statistical models
        assert len(result["errors"]) == 0

    def test_execute_with_min_models_requirement(self, sample_df, config):
        """Execute plan with min_models_for_ensemble requirement."""
        dataset = TSDataset.from_dataframe(sample_df, config)
        plan = Plan(
            statistical_models=[ModelCandidate(name="Naive")],
            min_models_for_ensemble=5,  # More than we have
        )

        with pytest.raises(EModelFailed, match="Insufficient models"):
            plan.execute(dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
