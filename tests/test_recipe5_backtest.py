"""Recipe 5: Backtest Analysis - End-to-end verification."""

import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, rolling_backtest
from tsagentkit.models import fit, predict
from tsagentkit.router import make_plan
from tsagentkit.series import TSDataset
from tsagentkit.qa import run_qa


def test_recipe5_backtest():
    """Test Recipe 5: Backtest Analysis."""
    print("=" * 60)
    print("RECIPE 5: Backtest Analysis")
    print("=" * 60)

    # Generate data with trend
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 60,
            "ds": pd.date_range("2024-01-01", periods=60, freq="D"),
            "y": np.linspace(100, 200, 60) + np.random.normal(0, 5, 60),
        }
    )
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Create dataset and plan
    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)
    qa_report = run_qa(df, spec, mode="standard")
    plan = make_plan(dataset, spec, qa_report)

    print(f"\n=== Plan ===")
    print(f"Primary model: {plan.primary_model}")
    print(f"Fallback chain: {plan.fallback_chain}")

    # Create wrapper functions for backtest (expects DataFrame input)
    def bt_fit(model_name: str, train_df: pd.DataFrame, config: dict):
        train_ds = TSDataset.from_dataframe(train_df, spec, validate=False)
        return fit(model_name, train_ds, config)

    def bt_predict(model, test_df: pd.DataFrame, horizon: int):
        test_ds = TSDataset.from_dataframe(test_df, spec, validate=False, compute_sparsity=False)
        return predict(model, test_ds, horizon)

    # Run detailed backtest
    print("\n=== Running Backtest ===")
    report = rolling_backtest(
        dataset=dataset,
        spec=spec,
        plan=plan,
        fit_func=bt_fit,
        predict_func=bt_predict,
        n_windows=3,  # Reduced for faster test
        window_strategy="expanding",
    )

    # Analyze results
    print("\n=== Aggregate Metrics ===")
    for metric, value in sorted(report.aggregate_metrics.items()):
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}")

    print("\n=== Per-Series Performance ===")
    for uid, metrics in report.series_metrics.items():
        wape = metrics.metrics.get("wape", 0)
        print(f"  {uid}: WAPE={wape:.2%}, windows={metrics.num_windows}")

    print("\n=== Window Results ===")
    for window in report.window_results:
        print(
            f"  Window {window.window_index}: "
            f"train={window.train_start} to {window.train_end}, "
            f"test={window.test_start} to {window.test_end}"
        )

    print("\n=== Report Summary ===")
    print(report.summary())

    # Assertions
    assert report is not None, "Report should not be None"
    assert report.n_windows > 0, "Should have backtest windows"
    assert len(report.window_results) > 0, "Should have window results"
    assert len(report.aggregate_metrics) > 0, "Should have aggregate metrics"
    assert "wape" in report.aggregate_metrics, "Should have WAPE metric"
    assert "smape" in report.aggregate_metrics, "Should have SMAPE metric"
    assert "mase" in report.aggregate_metrics, "Should have MASE metric"

    # Verify no errors occurred
    if report.errors:
        print(f"\n⚠️  Backtest errors: {report.errors}")
    assert len(report.errors) == 0, "Should have no backtest errors"

    print("\n✅ Recipe 5 PASSED")
    return True


if __name__ == "__main__":
    test_recipe5_backtest()
