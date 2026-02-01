"""Recipe 3: Intermittent Demand - End-to-end verification."""

import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast
from tsagentkit.router import make_plan
from tsagentkit.series import TSDataset


def generate_intermittent_data(n_parts=3, n_weeks=52) -> pd.DataFrame:
    """Generate intermittent demand (many zeros, occasional spikes)."""
    np.random.seed(456)

    records = []
    for part_id in range(n_parts):
        for week in range(n_weeks):
            # 70% chance of zero demand
            if np.random.random() < 0.7:
                demand = 0.0
            else:
                # Occasional demand spike
                demand = float(np.random.poisson(5) + 1)

            records.append(
                {
                    "unique_id": f"part_{part_id}",
                    "ds": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=week),
                    "y": demand,
                }
            )

    return pd.DataFrame(records)


def test_recipe3_intermittent():
    """Test Recipe 3: Intermittent Demand."""
    print("=" * 60)
    print("RECIPE 3: Intermittent Demand")
    print("=" * 60)

    # Load data
    df = generate_intermittent_data()
    print(f"Data shape: {df.shape}")

    # Check zero ratio
    print("\n=== Zero Density Analysis ===")
    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]
        zero_ratio = (series["y"] == 0).mean()
        print(f"{uid}: {zero_ratio:.1%} zeros")

    # Create task (Weekly, 4 weeks ahead)
    spec = TaskSpec(horizon=4, freq="W")

    # Build dataset and check sparsity classification
    print("\n=== Sparsity Classification ===")
    dataset = TSDataset.from_dataframe(df, spec)
    if dataset.sparsity_profile and dataset.sparsity_profile.series_profiles:
        for uid, metrics in dataset.sparsity_profile.series_profiles.items():
            classification = metrics.get("classification", "unknown")
            zero_ratio = metrics.get("zero_ratio", 0)
            print(f"{uid}: {classification} (zero_ratio: {zero_ratio:.2%})")

    # Check plan routing
    print("\n=== Plan Routing ===")
    from tsagentkit.qa import run_qa

    qa_report = run_qa(df, spec, mode="standard")
    plan = make_plan(dataset, spec, qa_report)
    print(f"Primary model: {plan.primary_model}")
    print(f"Fallback chain: {plan.fallback_chain}")

    # Run forecast
    print("\n=== Running Forecast ===")
    result = run_forecast(df, spec, mode="standard")

    print("\n=== Forecast ===")
    print(result.forecast.df)

    print("\n=== Model Selected ===")
    print(f"Model: {result.forecast.model_name}")
    print("(Intermittent series use appropriate models)")

    print("\n=== Summary ===")
    print(result.summary())

    # Assertions
    assert result is not None, "Result should not be None"
    assert result.forecast is not None, "Forecast should not be None"
    assert len(result.forecast.df) == 4 * 3, "Should have 4 weeks * 3 parts = 12 rows"

    # Check that intermittent series are detected
    if dataset.sparsity_profile:
        intermittent_count = sum(
            1
            for uid, m in dataset.sparsity_profile.series_profiles.items()
            if m.get("classification") == "intermittent"
        )
        print(f"\nIntermittent series detected: {intermittent_count}")

    print("\n✅ Recipe 3 PASSED")
    return True


if __name__ == "__main__":
    test_recipe3_intermittent()
