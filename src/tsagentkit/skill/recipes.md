# tsagentkit Recipes

## What
Runnable, end-to-end examples for common forecasting scenarios.

## When
Use these as templates when building scripts or demos.

## Inputs
- `data`: pandas DataFrame with `unique_id`, `ds`, `y`
- `task_spec`: `TaskSpec`

## Workflow
- Pick a recipe, generate or load data, define `TaskSpec`, call `run_forecast`.

## Recipe 1: Retail Daily Sales

**Scenario**: Daily sales forecasting for multiple retail stores with seasonal patterns.

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast

# Generate sample retail data
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

        for i, (date, sale) in enumerate(zip(
            pd.date_range("2024-01-01", periods=n_days, freq="D"),
            sales
        )):
            records.append({
                "unique_id": f"store_{store_id}",
                "ds": date,
                "y": float(sale),
            })

    return pd.DataFrame(records)

# Load data
df = generate_retail_data()
print(f"Data shape: {df.shape}")
print(f"Series: {df['unique_id'].unique()}")

# Define forecasting task
spec = TaskSpec(
    horizon=14,        # Forecast 2 weeks ahead
    freq="D",          # Daily frequency
    quantiles=[0.1, 0.5, 0.9],  # Include prediction intervals
)

# Run forecast (quick mode for demo)
result = run_forecast(df, spec, mode="quick")

# Review results
print("\n=== Forecast ===")
print(result.forecast.head())

print("\n=== Model Used ===")
print(result.model_name)

print("\n=== Provenance ===")
print(f"Data signature: {result.provenance['data_signature']}")
print(f"Timestamp: {result.provenance['timestamp']}")

print("\n=== Summary ===")
print(result.summary())
```

## Recipe 2: Industrial Hourly Metrics

**Scenario**: Hourly equipment sensor readings with irregular gaps.

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast, TSDataset
from tsagentkit.series import compute_sparsity_profile

# Generate hourly sensor data with gaps
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
                records.append({
                    "unique_id": f"sensor_{sensor_id}",
                    "ds": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                    "y": float(value),
                })

    return pd.DataFrame(records)

# Load data
df = generate_sensor_data()

# Analyze sparsity
print("=== Sparsity Analysis ===")
dataset = TSDataset.from_dataframe(df, TaskSpec(horizon=24, freq="H"))
profile = dataset.sparsity_profile
for uid, metrics in profile.series_profiles.items():
    print(f"{uid}: {metrics['classification']} "
          f"(gaps: {metrics.get('gap_ratio', 0):.2%})")

# Forecast next 24 hours
spec = TaskSpec(horizon=24, freq="H")
result = run_forecast(df, spec, mode="standard")

print("\n=== Results ===")
print(result.summary())
```

## Recipe 3: Intermittent Demand

**Scenario**: Spare parts demand with many zero values (intermittent demand).

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast
from tsagentkit.router import make_plan

# Generate intermittent demand data
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

            records.append({
                "unique_id": f"part_{part_id}",
                "ds": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=week),
                "y": demand,
            })

    return pd.DataFrame(records)

# Load data
df = generate_intermittent_data()

# Check zero ratio
for uid in df["unique_id"].unique():
    series = df[df["unique_id"] == uid]
    zero_ratio = (series["y"] == 0).mean()
    print(f"{uid}: {zero_ratio:.1%} zeros")

# Create task
spec = TaskSpec(horizon=4, freq="W")  # Weekly, 4 weeks ahead

# Run forecast
result = run_forecast(df, spec, mode="standard")

print("\n=== Forecast ===")
print(result.forecast)

print("\n=== Model Selected ===")
print(result.model_name)
print("(Intermittent series use appropriate models)")
```

## Recipe 4: Custom Model Integration

**Scenario**: Using a custom model with tsagentkit's pipeline.

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, rolling_backtest, run_forecast
from tsagentkit.contracts import ModelArtifact
from tsagentkit.series import TSDataset

# Define custom naive model
class NaiveModel:
    """Simple naive forecast model."""

    def __init__(self, season_length: int = 1):
        self.season_length = season_length
        self.last_values = {}

    def fit(self, df: pd.DataFrame) -> "NaiveModel":
        """Fit by storing last value per series."""
        for uid in df["unique_id"].unique():
            series = df[df["unique_id"] == uid].sort_values("ds")
            self.last_values[uid] = series["y"].iloc[-self.season_length:].values
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Generate naive forecast."""
        predictions = []
        for uid, values in self.last_values.items():
            for h in range(1, horizon + 1):
                # Cycle through last values
                idx = (h - 1) % len(values)
                predictions.append({
                    "unique_id": uid,
                    "yhat": values[idx],
                })
        return pd.DataFrame(predictions)

# Custom fit function
def custom_fit(model_name: str, data: pd.DataFrame, config: dict):
    """Fit custom model."""
    season_length = config.get("season_length", 1)
    model = NaiveModel(season_length=season_length)
    model.fit(data)

    return ModelArtifact(
        model=model,
        model_name=model_name,
        config=config,
    )

# Custom predict function
def custom_predict(model: ModelArtifact, data: pd.DataFrame, horizon: int):
    """Generate predictions."""
    naive_model = model.model
    preds = naive_model.predict(horizon)

    # Add dates
    last_dates = data.groupby("unique_id")["ds"].max()
    result_rows = []
    for _, row in preds.iterrows():
        uid = row["unique_id"]
        for h in range(1, horizon + 1):
            result_rows.append({
                "unique_id": uid,
                "ds": last_dates[uid] + pd.Timedelta(days=h),
                "yhat": row["yhat"],
            })

    return pd.DataFrame(result_rows)

# Generate data
df = pd.DataFrame({
    "unique_id": ["A"] * 30 + ["B"] * 30,
    "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
    "y": list(range(30)) * 2,
})

# Run with custom model
spec = TaskSpec(horizon=7, freq="D")
result = run_forecast(
    df, spec,
    mode="standard",
    fit_func=custom_fit,
    predict_func=custom_predict,
)

print("=== Custom Model Results ===")
print(result.summary())
```

## Recipe 5: Backtest Analysis

**Scenario**: Detailed backtest analysis to evaluate model performance.

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, rolling_backtest
from tsagentkit.models import fit, predict
from tsagentkit.router import make_plan
from tsagentkit.series import TSDataset

# Generate data with trend
df = pd.DataFrame({
    "unique_id": ["A"] * 60,
    "ds": pd.date_range("2024-01-01", periods=60, freq="D"),
    "y": np.linspace(100, 200, 60) + np.random.normal(0, 5, 60),
})

# Create dataset and plan
spec = TaskSpec(horizon=7, freq="D")
dataset = TSDataset.from_dataframe(df, spec)
plan = make_plan(dataset, spec)

# Run detailed backtest
report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit,
    predict_func=predict,
    n_windows=5,
    window_strategy="expanding",
)

# Analyze results
print("=== Aggregate Metrics ===")
for metric, value in sorted(report.aggregate_metrics.items()):
    print(f"{metric}: {value:.4f}")

print("\n=== Per-Series Performance ===")
for uid, metrics in report.series_metrics.items():
    print(f"{uid}: WAPE={metrics.metrics.get('wape', 0):.2%}, "
          f"windows={metrics.num_windows}")

print("\n=== Window Results ===")
for window in report.window_results:
    print(f"Window {window.window_index}: "
          f"train={window.train_start} to {window.train_end}, "
          f"test={window.test_start} to {window.test_end}")

print("\n=== Report Summary ===")
print(report.summary())
```

## Recipe 6: Error Handling

**Scenario**: Proper error handling and recovery.

```python
import pandas as pd
from tsagentkit import TaskSpec, run_forecast, validate_contract
from tsagentkit.contracts import (
    TSAgentKitError,
    EContractMissingColumn,
    ESplitRandomForbidden,
)

# Example 1: Handle missing columns
def safe_forecast_with_validation(df, spec):
    """Run forecast with proper validation."""
    # Validate first
    validation = validate_contract(df)
    if not validation.valid:
        print("Validation failed:")
        for error in validation.errors:
            print(f"  - {error['code']}: {error['message']}")

        # Auto-fix if possible
        if any(e['code'] == EContractMissingColumn.error_code for e in validation.errors):
            print("Hint: Ensure DataFrame has 'unique_id', 'ds', and 'y' columns")
        return None

    return run_forecast(df, spec)

# Example 2: Handle shuffled data
def safe_forecast_sorted(df, spec):
    """Ensure data is sorted before forecasting."""
    # Sort data to prevent E_SPLIT_RANDOM_FORBIDDEN
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return run_forecast(df, spec)

# Example 3: Comprehensive error handling
def robust_forecast(df, spec):
    """Run forecast with comprehensive error handling."""
    try:
        result = run_forecast(df, spec, mode="standard")
        print("Success!")
        return result

    except EContractMissingColumn as e:
        print(f"Data error: {e.message}")
        print(f"Available columns: {e.context.get('available', [])}")

    except ESplitRandomForbidden as e:
        print(f"Ordering error: {e.message}")
        print(f"Suggestion: {e.context.get('suggestion', '')}")
        # Try to fix
        df_sorted = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        print("Retrying with sorted data...")
        return robust_forecast(df_sorted, spec)

    except TSAgentKitError as e:
        print(f"tsagentkit error ({e.error_code}): {e.message}")

    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

    return None

# Test with various problematic inputs
print("=== Test 1: Valid Data ===")
df_valid = pd.DataFrame({
    "unique_id": ["A", "A", "B", "B"],
    "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
    "y": [1.0, 2.0, 3.0, 4.0],
})
spec = TaskSpec(horizon=2, freq="D")
result = robust_forecast(df_valid, spec)

print("\n=== Test 2: Missing Column ===")
df_missing = pd.DataFrame({"x": [1, 2, 3]})
robust_forecast(df_missing, spec)

print("\n=== Test 3: Shuffled Data ===")
df_shuffled = df_valid.sample(frac=1).reset_index(drop=True)
robust_forecast(df_shuffled, spec)
```

## Running the Recipes

All recipes are self-contained and can be run directly:

```bash
# Retail daily sales
python -c "$(cat recipe1_retail_daily.py)"

# Or save to file and run
python recipe1_retail_daily.py
```

## Next Steps

- See `README.md` for detailed module documentation
- Check `docs/PRD.md` for technical requirements
- Review test files in `tests/` for more examples
