"""Recipe 2: Industrial Hourly Metrics - End-to-end verification."""

import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast, TSDataset
from tsagentkit.series import compute_sparsity_profile


def generate_sensor_data(n_sensors=2, hours=168) -> pd.DataFrame:  # 1 week
    """Generate sensor data with some missing hours."""
    np.random.seed(123)

    records = []
    for sensor_id in range(n_sensors):
        # Base signal with daily pattern
        base = 50 + sensor_id * 10
        daily_pattern = 10 * np.sin(2 * np.pi * np.arange(hours) / 24)
        noise = np.random.normal(0, 2, hours)
        values = base + daily_pattern + noise

        # Random gaps (5% missing)
        gap_indices = np.random.choice(hours, size=int(hours * 0.05), replace=False)

        for hour, value in enumerate(values):
            if hour not in gap_indices:
                records.append(
                    {
                        "unique_id": f"sensor_{sensor_id}",
                        "ds": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                        "y": float(value),
                    }
                )

    return pd.DataFrame(records)


def test_recipe2_industrial_hourly():
    """Test Recipe 2: Industrial Hourly Metrics."""
    print("=" * 60)
    print("RECIPE 2: Industrial Hourly Metrics")
    print("=" * 60)

    # Load data
    df = generate_sensor_data()
    print(f"Data shape: {df.shape}")
    print(f"Series: {df['unique_id'].unique().tolist()}")

    # Analyze sparsity
    print("\n=== Sparsity Analysis ===")
    spec = TaskSpec(horizon=24, freq="h")  # Using "h" for hourly (pandas modern convention)
    dataset = TSDataset.from_dataframe(df, spec)
    profile = dataset.sparsity_profile

    if profile and profile.series_profiles:
        for uid, metrics in profile.series_profiles.items():
            classification = metrics.get("classification", "unknown")
            gap_ratio = metrics.get("gap_ratio", 0)
            print(f"{uid}: {classification} (gaps: {gap_ratio:.2%})")
    else:
        print("No sparsity profile available")

    # Forecast next 24 hours
    print("\n=== Running Forecast ===")
    result = run_forecast(df, spec, mode="standard")

    print("\n=== Results ===")
    print(result.summary())

    print("\n=== Forecast Sample ===")
    print(result.forecast.df.head(10))

    # Assertions
    assert result is not None, "Result should not be None"
    assert result.forecast is not None, "Forecast should not be None"
    assert len(result.forecast.df) > 0, "Forecast should have rows"
    assert result.backtest_report is not None, "Backtest should exist in standard mode"

    # Verify backtest was performed
    if result.backtest_report:
        n_windows = result.backtest_report.get("n_windows", 0)
        print(f"\nBacktest windows: {n_windows}")
        assert n_windows > 0, "Should have backtest windows"

    print("\n✅ Recipe 2 PASSED")
    return True


if __name__ == "__main__":
    test_recipe2_industrial_hourly()
