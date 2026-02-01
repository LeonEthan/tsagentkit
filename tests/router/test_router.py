"""Tests for router/router.py."""

import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.router import get_model_for_series, make_plan
from tsagentkit.series import TSDataset, SparsityProfile, compute_sparsity_profile


class TestMakePlan:
    """Tests for make_plan function."""

    @pytest.fixture
    def regular_dataset(self) -> TSDataset:
        """Create a regular dataset."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [1.0] * 20,
        })
        spec = TaskSpec(horizon=7, freq="D")
        return TSDataset.from_dataframe(df, spec)

    @pytest.fixture
    def intermittent_dataset(self) -> TSDataset:
        """Create an intermittent dataset."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 20,
            "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
            "y": [0] * 10 + [1] * 10,  # 50% zeros
        })
        spec = TaskSpec(horizon=7, freq="D")
        return TSDataset.from_dataframe(df, spec)

    @pytest.fixture
    def cold_start_dataset(self) -> TSDataset:
        """Create a cold-start dataset."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": [1.0] * 5,
        })
        spec = TaskSpec(horizon=7, freq="D")
        return TSDataset.from_dataframe(df, spec)

    def test_auto_strategy_regular(self, regular_dataset: TSDataset) -> None:
        """Test auto strategy for regular series."""
        plan = make_plan(regular_dataset, regular_dataset.task_spec, strategy="auto")
        assert plan.primary_model == "SeasonalNaive"
        assert plan.strategy == "auto"
        assert len(plan.fallback_chain) > 0

    def test_auto_strategy_intermittent(self, intermittent_dataset: TSDataset) -> None:
        """Test auto strategy for intermittent series."""
        plan = make_plan(intermittent_dataset, intermittent_dataset.task_spec, strategy="auto")
        # Should use seasonal naive for intermittent in v0.1
        assert plan.primary_model == "SeasonalNaive"

    def test_auto_strategy_cold_start(self, cold_start_dataset: TSDataset) -> None:
        """Test auto strategy for cold-start series."""
        plan = make_plan(cold_start_dataset, cold_start_dataset.task_spec, strategy="auto")
        assert plan.primary_model == "HistoricAverage"

    def test_baseline_only_strategy(self, regular_dataset: TSDataset) -> None:
        """Test baseline_only strategy."""
        plan = make_plan(regular_dataset, regular_dataset.task_spec, strategy="baseline_only")
        assert plan.strategy == "baseline_only"
        assert plan.primary_model == "SeasonalNaive"

    def test_tsfm_strategy_falls_back_to_auto(self, regular_dataset: TSDataset) -> None:
        """Test that tsfm_first falls back to auto when no TSFMs available."""
        plan = make_plan(regular_dataset, regular_dataset.task_spec, strategy="tsfm_first")
        # When no TSFMs are available, falls back to auto strategy
        assert plan.strategy == "auto"

    def test_plan_config_includes_season_length(self, regular_dataset: TSDataset) -> None:
        """Test that plan config includes season length."""
        plan = make_plan(regular_dataset, regular_dataset.task_spec)
        assert "season_length" in plan.config
        assert plan.config["season_length"] == 7  # Daily -> weekly

    def test_plan_config_includes_horizon(self, regular_dataset: TSDataset) -> None:
        """Test that plan config includes horizon."""
        plan = make_plan(regular_dataset, regular_dataset.task_spec)
        assert "horizon" in plan.config
        assert plan.config["horizon"] == 7

    def test_plan_config_with_quantiles(self, regular_dataset: TSDataset) -> None:
        """Test that plan config includes quantiles if specified."""
        spec = TaskSpec(horizon=7, freq="D", quantiles=[0.1, 0.9])
        plan = make_plan(regular_dataset, spec)
        assert "quantiles" in plan.config
        assert plan.config["quantiles"] == [0.1, 0.9]

    def test_unknown_strategy_raises(self, regular_dataset: TSDataset) -> None:
        """Test that unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            make_plan(regular_dataset, regular_dataset.task_spec, strategy="unknown")

    def test_plan_has_signature(self, regular_dataset: TSDataset) -> None:
        """Test that generated plan has signature."""
        plan = make_plan(regular_dataset, regular_dataset.task_spec)
        assert plan.signature != ""
        assert len(plan.signature) == 16


class TestGetModelForSeries:
    """Tests for get_model_for_series function."""

    @pytest.fixture
    def sparsity_profile(self) -> SparsityProfile:
        """Create a sparsity profile with different series types."""
        df = pd.DataFrame({
            "unique_id": ["regular"] * 20 + ["intermittent"] * 20 + ["cold"] * 5,
            "ds": list(pd.date_range("2024-01-01", periods=20, freq="D")) * 2 +
                  list(pd.date_range("2024-03-01", periods=5, freq="D")),
            "y": [1.0] * 20 + [0] * 10 + [1] * 10 + [1.0] * 5,
        })
        return compute_sparsity_profile(df, min_observations=10)

    def test_regular_series(self, sparsity_profile: SparsityProfile) -> None:
        """Test model selection for regular series."""
        model = get_model_for_series("regular", sparsity_profile)
        assert model == "SeasonalNaive"

    def test_intermittent_series(self, sparsity_profile: SparsityProfile) -> None:
        """Test model selection for intermittent series."""
        model = get_model_for_series("intermittent", sparsity_profile)
        assert model == "SeasonalNaive"

    def test_cold_start_series(self, sparsity_profile: SparsityProfile) -> None:
        """Test model selection for cold-start series."""
        model = get_model_for_series("cold", sparsity_profile)
        assert model == "HistoricAverage"

    def test_no_sparsity_profile(self) -> None:
        """Test model selection when no profile available."""
        model = get_model_for_series("any_series", None)
        assert model == "SeasonalNaive"

    def test_default_override(self, sparsity_profile: SparsityProfile) -> None:
        """Test using custom default model."""
        model = get_model_for_series("regular", sparsity_profile, default_model="CustomModel")
        assert model == "CustomModel"
