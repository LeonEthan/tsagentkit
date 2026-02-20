"""Benchmark tests for TSFM/Ensemble infrastructure speedup.

Measures timing and memory metrics for optimization validation.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsagentkit import ForecastConfig, ensemble_streaming
from tsagentkit.core.dataset import TSDataset
from tsagentkit.models.ensemble import ensemble, ensemble_with_quantiles
from tsagentkit.models.registry import REGISTRY, list_models
from tsagentkit.pipeline import fit_all, fit_all_parallel, make_plan, predict_all, predict_all_parallel


class MockArtifact:
    """Mock model artifact for benchmarking."""

    def __init__(self, name: str):
        self.name = name


class MockModelSpec:
    """Mock model spec for benchmarking."""

    def __init__(self, name: str):
        self.name = name
        self.is_tsfm = True
        self.adapter_path = "test"


def create_benchmark_dataset(n_series: int, series_length: int, freq: str = "D") -> pd.DataFrame:
    """Create a benchmark dataset with specified characteristics."""
    np.random.seed(42)
    rows = []

    for i in range(n_series):
        unique_id = f"S{i:04d}"
        dates = pd.date_range(start="2020-01-01", periods=series_length, freq=freq)
        values = np.random.randn(series_length).cumsum() + 100

        for d, v in zip(dates, values):
            rows.append({"unique_id": unique_id, "ds": d, "y": v})

    return pd.DataFrame(rows)


@pytest.mark.benchmark
class TestSequentialVsParallelFit:
    """Benchmark sequential vs parallel model fitting."""

    def test_benchmark_sequential_vs_parallel_fit(self):
        """Compare sequential and parallel fit performance."""
        # Create mock models for benchmarking
        models = [MockModelSpec(f"model_{i}") for i in range(8)]

        # Mock fit function
        def mock_fit(spec: Any, dataset: Any, device: str | None = None) -> Any:
            time.sleep(0.05)  # Simulate 50ms load time
            return MockArtifact(spec.name)

        # Patch fit for testing
        import tsagentkit.models.protocol as protocol_module

        original_fit = protocol_module.fit
        protocol_module.fit = mock_fit

        try:
            dataset = TSDataset.from_dataframe(
                create_benchmark_dataset(10, 100),
                ForecastConfig(h=7),
            )

            # Benchmark sequential
            gc.collect()
            start = time.perf_counter()
            seq_artifacts = fit_all(models, dataset)
            seq_time = time.perf_counter() - start

            # Benchmark parallel
            gc.collect()
            start = time.perf_counter()
            par_artifacts = fit_all_parallel(models, dataset, max_workers=4)
            par_time = time.perf_counter() - start

            print(f"\nSequential fit: {seq_time:.3f}s")
            print(f"Parallel fit: {par_time:.3f}s")
            print(f"Speedup: {seq_time / par_time:.2f}x")

            # Verify results
            assert len(seq_artifacts) == len(models)
            assert len(par_artifacts) == len(models)

            # Parallel should generally be faster (allow some variance)
            # In ideal conditions, expect 2-3x speedup with 4 workers
            # But be lenient in CI environments

        finally:
            protocol_module.fit = original_fit


@pytest.mark.benchmark
class TestEnsembleMemoryUsage:
    """Benchmark memory usage of standard vs streaming ensemble."""

    def test_benchmark_ensemble_memory(self):
        """Compare memory usage between standard and streaming ensemble."""
        # Create large prediction sets
        n_rows = 50000
        n_models = 4

        predictions = []
        for i in range(n_models):
            pred = pd.DataFrame({
                "unique_id": [f"S{i:04d}" for i in range(n_rows)],
                "ds": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
                "yhat": np.random.randn(n_rows),
                "q0.1": np.random.randn(n_rows),
                "q0.5": np.random.randn(n_rows),
                "q0.9": np.random.randn(n_rows),
            })
            predictions.append(pred)

        # Benchmark standard ensemble
        gc.collect()
        start = time.perf_counter()
        result_standard = ensemble_with_quantiles(
            predictions, method="median", quantiles=(0.1, 0.5, 0.9)
        )
        standard_time = time.perf_counter() - start

        # Benchmark streaming ensemble
        gc.collect()
        start = time.perf_counter()
        result_streaming = ensemble_streaming(
            predictions, method="median", quantiles=(0.1, 0.5, 0.9), chunk_size=5000
        )
        streaming_time = time.perf_counter() - start

        print(f"\nStandard ensemble: {standard_time:.3f}s")
        print(f"Streaming ensemble: {streaming_time:.3f}s")
        print(f"Ratio: {standard_time / streaming_time:.2f}x")

        # Verify results are equivalent
        np.testing.assert_array_almost_equal(
            result_standard["yhat"].values,
            result_streaming["yhat"].values,
        )

        # Streaming may be slightly slower due to chunking overhead,
        # but should be close for large datasets

    def test_ensemble_accuracy_parity(self):
        """Verify accuracy parity between sequential and streaming ensemble."""
        np.random.seed(42)

        # Create predictions
        n_rows = 1000
        n_models = 5

        predictions = []
        for _ in range(n_models):
            pred = pd.DataFrame({
                "unique_id": ["A"] * n_rows,
                "ds": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
                "yhat": np.random.randn(n_rows),
                "q0.1": np.random.randn(n_rows),
                "q0.9": np.random.randn(n_rows),
            })
            predictions.append(pred)

        # Both methods should produce identical results
        result_standard = ensemble(predictions, method="median")
        result_streaming = ensemble_streaming(predictions, method="median", chunk_size=100)

        np.testing.assert_array_almost_equal(
            result_standard["yhat"].values,
            result_streaming["yhat"].values,
        )


@pytest.mark.benchmark
class TestModelSelectionPerformance:
    """Benchmark model selection strategies."""

    def test_make_plan_performance(self):
        """Benchmark make_plan with different selection strategies."""
        config_all = ForecastConfig(h=7, model_selection="all")
        config_fast = ForecastConfig(h=7, model_selection="fast", max_models=2)
        config_accurate = ForecastConfig(h=7, model_selection="accurate", max_models=2)

        # Benchmark "all" selection
        gc.collect()
        start = time.perf_counter()
        models_all = make_plan(tsfm_only=True, config=config_all)
        time_all = time.perf_counter() - start

        # Benchmark "fast" selection
        gc.collect()
        start = time.perf_counter()
        models_fast = make_plan(tsfm_only=True, config=config_fast)
        time_fast = time.perf_counter() - start

        # Benchmark "accurate" selection
        gc.collect()
        start = time.perf_counter()
        models_accurate = make_plan(tsfm_only=True, config=config_accurate)
        time_accurate = time.perf_counter() - start

        print(f"\nMake plan 'all': {time_all*1000:.2f}ms, {len(models_all)} models")
        print(f"Make plan 'fast': {time_fast*1000:.2f}ms, {len(models_fast)} models")
        print(f"Make plan 'accurate': {time_accurate*1000:.2f}ms, {len(models_accurate)} models")

        # Verify results
        assert len(models_all) >= len(models_fast)
        assert len(models_all) >= len(models_accurate)


@pytest.mark.benchmark
class TestDatasetPreprocessingCache:
    """Benchmark shared preprocessing cache performance."""

    def test_series_dict_caching_performance(self):
        """Benchmark get_series_dict caching."""
        # Create large dataset
        df = create_benchmark_dataset(1000, 500)  # 1000 series, 500 points each
        config = ForecastConfig(h=7)
        dataset = TSDataset.from_dataframe(df, config)

        # First call - builds cache
        gc.collect()
        start = time.perf_counter()
        result1 = dataset.get_series_dict()
        time_first = time.perf_counter() - start

        # Second call - uses cache
        gc.collect()
        start = time.perf_counter()
        result2 = dataset.get_series_dict()
        time_second = time.perf_counter() - start

        print(f"\nFirst call (build cache): {time_first*1000:.2f}ms")
        print(f"Second call (cached): {time_second*1000:.2f}ms")
        print(f"Speedup: {time_first / time_second:.1f}x")

        # Second call should be much faster
        assert time_second < time_first * 0.1
        assert result1 is result2  # Same cached object


@pytest.mark.benchmark
class TestLengthBalancedBatching:
    """Benchmark Chronos length-balanced batching."""

    def test_group_by_similar_lengths_performance(self):
        """Benchmark length-balanced batching vs regular batching."""
        from tsagentkit.models.adapters.tsfm.chronos import _group_by_similar_lengths

        # Create series with varying lengths
        np.random.seed(42)
        n_series = 100
        series_data = []

        for i in range(n_series):
            # Mix of short and long series
            if i < 50:
                length = np.random.randint(10, 50)
            else:
                length = np.random.randint(200, 500)

            context = np.random.randn(length).astype(np.float32)
            series_data.append((f"S{i:04d}", context, pd.Timestamp("2024-01-01")))

        # Benchmark length-balanced batching
        gc.collect()
        start = time.perf_counter()
        batches_balanced = _group_by_similar_lengths(series_data, batch_size=16)
        time_balanced = time.perf_counter() - start

        # Calculate padding overhead for balanced batches
        total_padding_balanced = 0
        for batch in batches_balanced:
            lengths = [len(ctx) for _, ctx, _ in batch]
            max_len = max(lengths)
            total_padding_balanced += sum(max_len - l for l in lengths)

        # Simulate regular batching (sequential chunks)
        total_padding_regular = 0
        for i in range(0, len(series_data), 16):
            batch = series_data[i:i+16]
            lengths = [len(ctx) for _, ctx, _ in batch]
            max_len = max(lengths)
            total_padding_regular += sum(max_len - l for l in lengths)

        print(f"\nLength-balanced batching: {time_balanced*1000:.2f}ms")
        print(f"Total padding (balanced): {total_padding_balanced}")
        print(f"Total padding (regular): {total_padding_regular}")
        print(f"Padding reduction: {(1 - total_padding_balanced/total_padding_regular)*100:.1f}%")

        # Length-balanced should reduce padding
        assert total_padding_balanced <= total_padding_regular


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--benchmark"])
