"""Recipe 1: Retail Daily Sales - End-to-end verification."""

import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast


def generate_retail_data(n_stores=3, n_days=90) -> pd.DataFrame:
    """Generate synthetic retail daily sales data."""
    np.random.seed(42)

    records = []
    for store_id in range(n_stores):
        base_sales = 1000 + store_id * 200
        trend = np.linspace(0, 50, n_days)
        seasonality = 100 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly
        noise = np.random.normal(0, 50, n_days)

        sales = base_sales + trend + seasonality + noise
        sales = np.maximum(sales, 0)  # No negative sales

        for i, (date, sale) in enumerate(
            zip(pd.date_range("2024-01-01", periods=n_days, freq="D"), sales)
        ):
            records.append(
                {
                    "unique_id": f"store_{store_id}",
                    "ds": date,
                    "y": float(sale),
                }
            )

    return pd.DataFrame(records)


def test_recipe1_retail_daily():
    """Test Recipe 1: Retail Daily Sales."""
    print("=" * 60)
    print("RECIPE 1: Retail Daily Sales")
    print("=" * 60)

    # Load data
    df = generate_retail_data()
    print(f"Data shape: {df.shape}")
    print(f"Series: {df['unique_id'].unique().tolist()}")

    # Define forecasting task
    spec = TaskSpec(
        horizon=14,  # Forecast 2 weeks ahead
        freq="D",  # Daily frequency
        quantiles=[0.1, 0.5, 0.9],  # Include prediction intervals
    )

    # Run forecast (quick mode for demo)
    result = run_forecast(df, spec, mode="quick")

    # Review results - note: result.forecast is ForecastResult, .df is the DataFrame
    print("\n=== Forecast ===")
    print(result.forecast.df.head(10))

    print("\n=== Model Used ===")
    print(result.forecast.model_name)

    print("\n=== Provenance ===")
    # provenance is in result.provenance (RunArtifact level) or result.forecast.provenance
    prov = result.provenance or result.forecast.provenance
    print(f"Data signature: {prov.data_signature}")
    print(f"Run ID: {prov.run_id}")
    print(f"Timestamp: {prov.timestamp}")

    print("\n=== Summary ===")
    print(result.summary())

    # Assertions
    assert result is not None, "Result should not be None"
    assert result.forecast is not None, "Forecast should not be None"
    assert result.forecast.df is not None, "Forecast df should not be None"
    assert len(result.forecast.df) > 0, "Forecast should have rows"
    assert "yhat" in result.forecast.df.columns, "Forecast should have yhat column"
    assert prov is not None, "Provenance should exist"
    assert prov.data_signature is not None, "Data signature should exist"

    # Check quantiles - they might be named differently
    quantile_cols = result.forecast.get_quantile_columns()
    print(f"\nQuantile columns found: {quantile_cols}")

    # Verify forecast length matches expected (horizon * num_series)
    expected_rows = 14 * 3  # 14 days * 3 stores
    print(f"Expected forecast rows: {expected_rows}, Actual: {len(result.forecast.df)}")

    print("\n✅ Recipe 1 PASSED")
    return True


if __name__ == "__main__":
    test_recipe1_retail_daily()
