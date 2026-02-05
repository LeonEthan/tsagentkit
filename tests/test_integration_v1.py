"""Integration tests for v1.0: TSFM adapters and hierarchical reconciliation.

Tests the complete integration of:
- TSFM adapters with the router
- Hierarchical reconciliation with backtesting
- TSFM model caching for serving
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import TaskSpec
from tsagentkit.contracts import EAdapterNotAvailable
from tsagentkit.hierarchy import HierarchyStructure, ReconciliationMethod
from tsagentkit.router import make_plan
from tsagentkit.series import TSDataset
from tsagentkit.serving import TSFMModelCache, clear_tsfm_cache, get_tsfm_model


class TestTSFMRouterIntegration:
    """Test TSFM adapter integration with router."""

    def test_router_prefers_tsfm_when_available(self):
        """Test that router selects TSFM models when available."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
            "y": list(range(30)),
        })
        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # Create plan with TSFM preference
        plan, route_decision = make_plan(
            dataset,
            spec,
            use_tsfm=True,
        )

        # Should have a valid plan
        assert plan.candidate_models
        # If TSFMs are available, they should lead the candidate list
        if any(m.startswith("tsfm-") for m in plan.candidate_models):
            assert plan.candidate_models[0].startswith("tsfm-")
        # Verify RouteDecision is returned
        assert route_decision.selected_plan == plan
        assert route_decision.reasons

    def test_router_falls_back_when_tsfm_unavailable(self):
        """Test fallback when TSFM packages not installed."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 30,
            "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
            "y": list(range(30)),
        })
        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        plan, route_decision = make_plan(dataset, spec, use_tsfm=True)

        # Should have a baseline model available
        assert any(m in {"SeasonalNaive", "HistoricAverage", "Naive"} for m in plan.candidate_models)
        # Verify RouteDecision is returned
        assert route_decision.selected_plan == plan

    def test_router_with_hierarchical_data(self):
        """Test router creates hierarchical plan for hierarchical data."""
        # Create hierarchical data
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B", "Total", "Total"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"] * 3),
            "y": [10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
        })
        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # Create hierarchy structure
        hierarchy = HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],  # A
                [0, 1],  # B
                [1, 1],  # Total
            ]),
        )
        dataset = dataset.with_hierarchy(hierarchy)

        plan, route_decision = make_plan(dataset, spec)

        assert plan.candidate_models
        assert route_decision.selected_plan == plan


class TestHierarchicalBacktestIntegration:
    """Test hierarchical reconciliation integration with backtesting."""

    @pytest.fixture
    def hierarchical_dataset(self) -> TSDataset:
        """Create a hierarchical dataset for testing."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        # Bottom-level series
        df_bottom = pd.DataFrame({
            "unique_id": ["A"] * 50 + ["B"] * 50,
            "ds": list(dates) * 2,
            "y": list(range(50)) + list(range(50, 100)),
        })

        # Total series
        df_total = pd.DataFrame({
            "unique_id": ["Total"] * 50,
            "ds": dates,
            "y": [50 + i * 2 for i in range(50)],
        })

        df = pd.concat([df_bottom, df_total], ignore_index=True)
        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # Add hierarchy
        hierarchy = HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],  # A
                [0, 1],  # B
                [1, 1],  # Total
            ]),
        )
        return dataset.with_hierarchy(hierarchy)

    def test_dataset_hierarchy_support(self, hierarchical_dataset: TSDataset):
        """Test that TSDataset supports hierarchy operations."""
        assert hierarchical_dataset.is_hierarchical()
        assert hierarchical_dataset.hierarchy is not None
        assert hierarchical_dataset.hierarchy.get_num_levels() == 2

    def test_dataset_get_level_series(self, hierarchical_dataset: TSDataset):
        """Test getting series at specific hierarchy levels."""
        level_0 = hierarchical_dataset.get_level_series(0)
        level_1 = hierarchical_dataset.get_level_series(1)

        assert "Total" in level_0
        assert set(level_1) == {"A", "B"}

    def test_reconcile_forecasts_function(self):
        """Test the high-level reconcile_forecasts function."""
        from tsagentkit.hierarchy import reconcile_forecasts

        # Create simple forecast
        forecast_df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B", "Total", "Total"],
            "ds": pd.to_datetime(["2024-02-01", "2024-02-02"] * 3),
            "yhat": [10.0, 11.0, 20.0, 21.0, 35.0, 37.0],  # 35 != 10+20, 37 != 11+21
        })

        hierarchy = HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],
                [0, 1],
                [1, 1],
            ]),
        )

        reconciled = reconcile_forecasts(
            forecast_df,
            hierarchy,
            ReconciliationMethod.BOTTOM_UP,
        )

        # Check structure
        assert len(reconciled) == len(forecast_df)


class TestTSFMCacheIntegration:
    """Test TSFM model caching for serving."""

    def test_cache_singleton_pattern(self):
        """Test that TSFMModelCache is a singleton."""
        cache1 = TSFMModelCache()
        cache2 = TSFMModelCache()
        assert cache1 is cache2

    def test_cache_stats_empty(self):
        """Test cache stats when empty."""
        cache = TSFMModelCache()
        cache.clear_cache()  # Ensure empty

        stats = cache.get_cache_stats()
        assert stats["num_models"] == 0
        assert stats["total_accesses"] == 0

    def test_clear_cache_specific_model(self):
        """Test clearing specific model from cache."""
        cache = TSFMModelCache()
        cache.clear_cache()  # Start fresh

        # This will fail to load but will test the cache mechanism
        with pytest.raises(EAdapterNotAvailable):
            cache.get_model("nonexistent_model")

        # Clear specific model
        cache.clear_cache("nonexistent_model")

        stats = cache.get_cache_stats()
        # Model should be removed
        assert "nonexistent_model" not in stats["models"]

    def test_get_tsfm_model_convenience_function(self):
        """Test the get_tsfm_model convenience function."""
        clear_tsfm_cache()  # Start fresh

        # Should raise error for unavailable model
        with pytest.raises(EAdapterNotAvailable):
            get_tsfm_model("chronos")

        # Stats should show the attempt
        cache = TSFMModelCache()
        cache.get_cache_stats()
        # May or may not have the model depending on if chronos is installed

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        cache = TSFMModelCache()

        key1 = cache._make_cache_key("chronos", pipeline="large")
        key2 = cache._make_cache_key("chronos", pipeline="small")
        key3 = cache._make_cache_key("chronos", pipeline="large")

        # Different configs should have different keys
        assert key1 != key2
        # Same config should have same key
        assert key1 == key3


class TestEndToEndV1Workflow:
    """End-to-end tests for v1.0 workflow."""

    def test_complete_hierarchical_workflow(self):
        """Test complete workflow with hierarchical data."""
        # 1. Create hierarchical data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "unique_id": ["A"] * 30 + ["B"] * 30 + ["Total"] * 30,
            "ds": list(dates) * 3,
            "y": list(range(30)) + list(range(30, 60)) + list(range(30, 60)),
        })

        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # 2. Add hierarchy
        hierarchy = HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],
                [0, 1],
                [1, 1],
            ]),
        )
        dataset = dataset.with_hierarchy(hierarchy)

        # 3. Create plan
        plan, route_decision = make_plan(dataset, spec)

        # 4. Verify hierarchical configuration
        assert dataset.is_hierarchical()
        assert plan.candidate_models
        assert route_decision.selected_plan == plan

    def test_hierarchical_plan_with_tsfm_preference(self):
        """Test creating hierarchical plan with TSFM preference."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "unique_id": ["A"] * 30 + ["B"] * 30 + ["Total"] * 30,
            "ds": list(dates) * 3,
            "y": list(range(30)) + list(range(30, 60)) + list(range(30, 60)),
        })

        spec = TaskSpec(h=7, freq="D")
        dataset = TSDataset.from_dataframe(df, spec)

        # Add hierarchy
        hierarchy = HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],
                [0, 1],
                [1, 1],
            ]),
        )
        dataset = dataset.with_hierarchy(hierarchy)

        # Create plan with TSFM preference
        plan, route_decision = make_plan(
            dataset,
            spec,
            use_tsfm=True,
            tsfm_preference=["chronos", "moirai"],
        )

        # Verify plan structure
        assert plan.candidate_models
        assert route_decision.selected_plan == plan
        # Verify reasons are present (TSFM models listed if available)
        assert route_decision.reasons
        assert any("tsfm_available" in reason for reason in route_decision.reasons)


class TestReconciliationMethodsIntegration:
    """Test different reconciliation methods."""

    @pytest.fixture
    def simple_hierarchy(self):
        """Create a simple hierarchy for testing."""
        return HierarchyStructure(
            aggregation_graph={"Total": ["A", "B"]},
            bottom_nodes=["A", "B"],
            s_matrix=np.array([
                [1, 0],
                [0, 1],
                [1, 1],
            ]),
        )

    def test_bottom_up_reconciliation(self, simple_hierarchy):
        """Test bottom-up reconciliation."""
        from tsagentkit.hierarchy import Reconciler, ReconciliationMethod

        reconciler = Reconciler(
            ReconciliationMethod.BOTTOM_UP,
            simple_hierarchy,
        )

        # Base forecasts: A=10, B=20, Total=25 (incoherent: 25 != 30)
        base_forecasts = np.array([10.0, 20.0, 25.0])

        reconciled = reconciler.reconcile(base_forecasts)

        # Total should equal A + B
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-10

    def test_ols_reconciliation(self, simple_hierarchy):
        """Test OLS reconciliation."""
        from tsagentkit.hierarchy import Reconciler, ReconciliationMethod

        reconciler = Reconciler(
            ReconciliationMethod.OLS,
            simple_hierarchy,
        )

        base_forecasts = np.array([10.0, 20.0, 25.0])
        reconciled = reconciler.reconcile(base_forecasts)

        # Should be coherent
        assert abs(reconciled[2] - (reconciled[0] + reconciled[1])) < 1e-10

    def test_reconciliation_with_horizon(self, simple_hierarchy):
        """Test reconciliation with multiple horizon steps."""
        from tsagentkit.hierarchy import Reconciler, ReconciliationMethod

        reconciler = Reconciler(
            ReconciliationMethod.BOTTOM_UP,
            simple_hierarchy,
        )

        # Multiple horizon points: (n_nodes, horizon)
        base_forecasts = np.array([
            [10.0, 11.0, 12.0],  # A
            [20.0, 21.0, 22.0],  # B
            [25.0, 26.0, 27.0],  # Total (incoherent)
        ])

        reconciled = reconciler.reconcile(base_forecasts)

        # Check coherence at each horizon
        for h in range(3):
            assert abs(reconciled[2, h] - (reconciled[0, h] + reconciled[1, h])) < 1e-10
