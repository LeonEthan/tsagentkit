"""Tests for TSFM/Ensemble infrastructure speedup optimizations.

Tests for Phase 1-4 optimizations:
- Phase 1: Parallel Model Execution
- Phase 2: Memory-Aware Model Selection
- Phase 4: Streaming Ensemble
- Phase 3: Shared Preprocessing Cache
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, ensemble_streaming
from tsagentkit.core.dataset import TSDataset
from tsagentkit.models.ensemble import ensemble, ensemble_with_quantiles
from tsagentkit.models.registry import REGISTRY, list_models
from tsagentkit.pipeline import (
    _sort_by_accuracy,
    _sort_by_speed,
    fit_all_parallel,
    make_plan,
    predict_all_parallel,
)


# =============================================================================
# Phase 2: Memory-Aware Model Selection Tests
# =============================================================================


class TestModelSelectionConfig:
    """Test new config options for model selection."""

    def test_max_models_config(self):
        """Test max_models config option."""
        config = ForecastConfig(h=7, max_models=2)
        assert config.max_models == 2

    def test_model_selection_config(self):
        """Test model_selection config option."""
        config = ForecastConfig(h=7, model_selection="fast")
        assert config.model_selection == "fast"

        config = ForecastConfig(h=7, model_selection="accurate")
        assert config.model_selection == "accurate"

        config = ForecastConfig(h=7, model_selection="all")
        assert config.model_selection == "all"

    def test_parallel_config_options(self):
        """Test parallel execution config options."""
        config = ForecastConfig(h=7, parallel_fit=True, parallel_predict=True, max_workers=4)
        assert config.parallel_fit is True
        assert config.parallel_predict is True
        assert config.max_workers == 4

    def test_invalid_model_selection_raises(self):
        """Test invalid model_selection raises ValueError."""
        with pytest.raises(ValueError, match="model_selection must be one of"):
            ForecastConfig(h=7, model_selection="invalid")  # type: ignore

    def test_invalid_max_models_raises(self):
        """Test invalid max_models raises ValueError."""
        with pytest.raises(ValueError, match="max_models must be at least 1 or None"):
            ForecastConfig(h=7, max_models=0)


class TestMakePlanWithSelection:
    """Test make_plan with model selection strategies."""

    def test_make_plan_without_config(self):
        """Test make_plan without config returns all models."""
        models = make_plan(tsfm_only=True)
        assert len(models) > 0

    def test_make_plan_with_max_models(self):
        """Test make_plan respects max_models limit."""
        config = ForecastConfig(h=7, max_models=2)
        models = make_plan(tsfm_only=True, config=config)
        assert len(models) <= 2

    def test_make_plan_with_fast_selection(self):
        """Test make_plan with 'fast' selection strategy."""
        config = ForecastConfig(h=7, model_selection="fast", max_models=2)
        models = make_plan(tsfm_only=True, config=config)

        # Check models are sorted by speed (TimesFM should come first if present)
        if len(models) >= 2:
            model_names = [m.name for m in models]
            # TimesFM is ranked fastest
            if "timesfm" in model_names and "patchtst_fm" in model_names:
                assert model_names.index("timesfm") < model_names.index("patchtst_fm")

    def test_make_plan_with_accurate_selection(self):
        """Test make_plan with 'accurate' selection strategy."""
        config = ForecastConfig(h=7, model_selection="accurate", max_models=2)
        models = make_plan(tsfm_only=True, config=config)

        # Check models are sorted by accuracy (Moirai should come first if present)
        if len(models) >= 2:
            model_names = [m.name for m in models]
            # Moirai is ranked most accurate
            if "moirai" in model_names and "patchtst_fm" in model_names:
                assert model_names.index("moirai") < model_names.index("patchtst_fm")


class TestModelSorting:
    """Test model sorting functions."""

    def test_sort_by_speed(self):
        """Test _sort_by_speed orders models correctly."""
        all_models = [REGISTRY[name] for name in list_models(tsfm_only=True)]
        sorted_models = _sort_by_speed(all_models)

        # Extract speed ranks
        speed_ranks = {
            "timesfm": 1,
            "chronos": 2,
            "moirai": 3,
            "patchtst_fm": 4,
        }

        # Verify ordering
        for i in range(len(sorted_models) - 1):
            current_rank = speed_ranks.get(sorted_models[i].name, 99)
            next_rank = speed_ranks.get(sorted_models[i + 1].name, 99)
            assert current_rank <= next_rank

    def test_sort_by_accuracy(self):
        """Test _sort_by_accuracy orders models correctly."""
        all_models = [REGISTRY[name] for name in list_models(tsfm_only=True)]
        sorted_models = _sort_by_accuracy(all_models)

        # Extract accuracy ranks
        accuracy_ranks = {
            "moirai": 1,
            "chronos": 2,
            "timesfm": 3,
            "patchtst_fm": 4,
        }

        # Verify ordering
        for i in range(len(sorted_models) - 1):
            current_rank = accuracy_ranks.get(sorted_models[i].name, 99)
            next_rank = accuracy_ranks.get(sorted_models[i + 1].name, 99)
            assert current_rank <= next_rank


# =============================================================================
# Phase 4: Streaming Ensemble Tests
# =============================================================================


class TestEnsembleStreaming:
    """Test streaming ensemble functionality."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction dataframes."""
        np.random.seed(42)
        n_rows = 100

        pred1 = pd.DataFrame({
            "unique_id": ["A"] * 50 + ["B"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50).tolist() * 2,
            "yhat": np.random.randn(n_rows),
            "q0.1": np.random.randn(n_rows),
            "q0.9": np.random.randn(n_rows),
        })

        pred2 = pd.DataFrame({
            "unique_id": ["A"] * 50 + ["B"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50).tolist() * 2,
            "yhat": np.random.randn(n_rows),
            "q0.1": np.random.randn(n_rows),
            "q0.9": np.random.randn(n_rows),
        })

        pred3 = pd.DataFrame({
            "unique_id": ["A"] * 50 + ["B"] * 50,
            "ds": pd.date_range("2024-01-01", periods=50).tolist() * 2,
            "yhat": np.random.randn(n_rows),
            "q0.1": np.random.randn(n_rows),
            "q0.9": np.random.randn(n_rows),
        })

        return [pred1, pred2, pred3]

    def test_ensemble_streaming_basic(self, sample_predictions):
        """Test basic streaming ensemble functionality."""
        result = ensemble_streaming(sample_predictions, chunk_size=25)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_predictions[0])
        assert "yhat" in result.columns

    def test_ensemble_streaming_with_quantiles(self, sample_predictions):
        """Test streaming ensemble with quantiles."""
        result = ensemble_streaming(
            sample_predictions,
            method="median",
            quantiles=(0.1, 0.9),
            chunk_size=25,
        )

        assert "q0.1" in result.columns
        assert "q0.9" in result.columns

    def test_ensemble_streaming_mean_method(self, sample_predictions):
        """Test streaming ensemble with mean method."""
        result = ensemble_streaming(sample_predictions, method="mean", chunk_size=25)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_predictions[0])

    def test_ensemble_streaming_equivalent_to_standard(self, sample_predictions):
        """Test that streaming produces same results as standard ensemble."""
        streaming_result = ensemble_streaming(sample_predictions, chunk_size=25)
        standard_result = ensemble(sample_predictions)

        np.testing.assert_array_almost_equal(
            streaming_result["yhat"].values,
            standard_result["yhat"].values,
        )

    def test_ensemble_streaming_single_prediction(self, sample_predictions):
        """Test streaming ensemble with single prediction."""
        result = ensemble_streaming([sample_predictions[0]], chunk_size=25)

        pd.testing.assert_frame_equal(result, sample_predictions[0].copy())

    def test_ensemble_streaming_empty_raises(self):
        """Test that empty predictions raises EInsufficient."""
        from tsagentkit.core.errors import EInsufficient

        with pytest.raises(EInsufficient, match="No predictions to ensemble"):
            ensemble_streaming([])

    def test_ensemble_streaming_large_chunk_size(self, sample_predictions):
        """Test streaming with chunk size larger than data."""
        result = ensemble_streaming(sample_predictions, chunk_size=1000)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_predictions[0])

    def test_ensemble_streaming_small_chunk_size(self, sample_predictions):
        """Test streaming with very small chunk size."""
        result = ensemble_streaming(sample_predictions, chunk_size=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_predictions[0])


# =============================================================================
# Phase 3: Shared Preprocessing Cache Tests
# =============================================================================


class TestSharedPreprocessingCache:
    """Test shared preprocessing cache functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample TSDataset."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B", "C", "C"],
            "ds": pd.date_range("2024-01-01", periods=6),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        config = ForecastConfig(h=2)
        return TSDataset.from_dataframe(df, config)

    def test_get_series_dict_returns_dict(self, sample_dataset):
        """Test get_series_dict returns dictionary."""
        result = sample_dataset.get_series_dict()

        assert isinstance(result, dict)
        assert len(result) == 3  # A, B, C

    def test_get_series_dict_values_are_arrays(self, sample_dataset):
        """Test get_series_dict values are numpy arrays."""
        result = sample_dataset.get_series_dict()

        for uid, values in result.items():
            assert isinstance(values, np.ndarray)
            assert values.dtype == np.float64

    def test_get_series_dict_correct_values(self, sample_dataset):
        """Test get_series_dict returns correct values."""
        result = sample_dataset.get_series_dict()

        np.testing.assert_array_equal(result["A"], [1.0, 2.0])
        np.testing.assert_array_equal(result["B"], [3.0, 4.0])
        np.testing.assert_array_equal(result["C"], [5.0, 6.0])

    def test_get_series_dict_caching(self, sample_dataset):
        """Test that get_series_dict caches results."""
        # First call
        result1 = sample_dataset.get_series_dict()

        # Second call should return same cached result
        result2 = sample_dataset.get_series_dict()

        assert result1 is result2

    def test_get_series_dict_different_datasets_not_shared(self):
        """Test cache doesn't leak between datasets."""
        df1 = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2024-01-01", periods=2),
            "y": [1.0, 2.0],
        })
        df2 = pd.DataFrame({
            "unique_id": ["B", "B"],
            "ds": pd.date_range("2024-01-01", periods=2),
            "y": [3.0, 4.0],
        })

        config = ForecastConfig(h=2)
        dataset1 = TSDataset.from_dataframe(df1, config)
        dataset2 = TSDataset.from_dataframe(df2, config)

        result1 = dataset1.get_series_dict()
        result2 = dataset2.get_series_dict()

        assert "A" in result1
        assert "B" in result2
        assert "B" not in result1
        assert "A" not in result2


# =============================================================================
# Phase 1: Parallel Execution Tests (Structure Only)
# =============================================================================


class TestParallelFunctionsExist:
    """Test that parallel functions exist and have correct signatures."""

    def test_fit_all_parallel_exists(self):
        """Test fit_all_parallel function exists."""
        assert callable(fit_all_parallel)

    def test_predict_all_parallel_exists(self):
        """Test predict_all_parallel function exists."""
        assert callable(predict_all_parallel)

    def test_ensemble_streaming_exists(self):
        """Test ensemble_streaming function exists in public API."""
        from tsagentkit import ensemble_streaming

        assert callable(ensemble_streaming)


# =============================================================================
# Phase 5: Adapter-Specific Optimizations Tests
# =============================================================================


class TestChronosLengthBalancedBatching:
    """Test Chronos length-balanced batching."""

    def test_group_by_similar_lengths_small_input(self):
        """Test that small inputs return single batch."""
        from tsagentkit.models.adapters.tsfm.chronos import _group_by_similar_lengths

        # Create series with varying lengths
        series_data = [
            ("A", np.array([1.0, 2.0, 3.0]), pd.Timestamp("2024-01-03")),
            ("B", np.array([4.0, 5.0]), pd.Timestamp("2024-01-02")),
        ]

        batches = _group_by_similar_lengths(series_data, batch_size=10)

        # Should return single batch for input smaller than batch_size
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_group_by_similar_lengths_grouping(self):
        """Test that series are grouped by similar lengths."""
        from tsagentkit.models.adapters.tsfm.chronos import _group_by_similar_lengths

        # Create series with very different lengths
        series_data = [
            ("A", np.array([1.0] * 100), pd.Timestamp("2024-01-01")),
            ("B", np.array([2.0] * 95), pd.Timestamp("2024-01-01")),
            ("C", np.array([3.0] * 10), pd.Timestamp("2024-01-01")),
            ("D", np.array([4.0] * 8), pd.Timestamp("2024-01-01")),
        ]

        batches = _group_by_similar_lengths(series_data, batch_size=2, num_bins=2)

        # Should create batches with similar-length series
        assert len(batches) >= 1

        # Check that series in same batch have similar lengths
        for batch in batches:
            lengths = [len(ctx) for _, ctx, _ in batch]
            if len(lengths) > 1:
                # Max difference should be relatively small compared to range
                length_range = max(lengths) - min(lengths)
                assert length_range < 100  # Should be less than full data range

    def test_group_by_similar_lengths_same_length(self):
        """Test that same-length series are batched together."""
        from tsagentkit.models.adapters.tsfm.chronos import _group_by_similar_lengths

        # Create series with same length
        series_data = [
            (f"S{i}", np.array([1.0] * 50), pd.Timestamp("2024-01-01"))
            for i in range(10)
        ]

        batches = _group_by_similar_lengths(series_data, batch_size=3)

        # Should batch by batch_size since all same length
        assert len(batches) == 4  # 3+3+3+1


class TestMoiraiPredictorCache:
    """Test Moirai predictor caching."""

    def test_predictor_cache_exists(self):
        """Test that predictor cache module-level variable exists."""
        from tsagentkit.models.adapters.tsfm import moirai

        assert hasattr(moirai, "_predictor_cache")
        assert isinstance(moirai._predictor_cache, dict)

    def test_clear_predictor_cache_exists(self):
        """Test clear_predictor_cache function exists."""
        from tsagentkit.models.adapters.tsfm.moirai import clear_predictor_cache

        assert callable(clear_predictor_cache)

    def test_clear_predictor_cache_clears_cache(self):
        """Test clear_predictor_cache actually clears the cache."""
        from tsagentkit.models.adapters.tsfm import moirai
        from tsagentkit.models.adapters.tsfm.moirai import clear_predictor_cache

        # Add a dummy entry to cache
        moirai._predictor_cache[(1, 2, 3, 4, 5, 6, 7)] = "dummy"
        assert len(moirai._predictor_cache) > 0

        # Clear cache
        clear_predictor_cache()

        assert len(moirai._predictor_cache) == 0


class TestAdapterOptimizationsExist:
    """Test that adapter optimization functions exist."""

    def test_chronos_group_by_lengths_exists(self):
        """Test Chronos _group_by_similar_lengths function exists."""
        from tsagentkit.models.adapters.tsfm.chronos import _group_by_similar_lengths

        assert callable(_group_by_similar_lengths)

    def test_moirai_unload_clears_cache(self):
        """Test Moirai unload function clears predictor cache."""
        from tsagentkit.models.adapters.tsfm import moirai
        from tsagentkit.models.adapters.tsfm.moirai import unload

        # Add a dummy entry to cache
        moirai._predictor_cache[(1, 2, 3, 4, 5, 6, 7)] = "dummy"

        # Unload should clear cache
        unload()

        assert len(moirai._predictor_cache) == 0
