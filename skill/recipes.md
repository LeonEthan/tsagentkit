# tsagentkit Recipes

> **Runnable Examples**: Complete workflows for common time-series forecasting scenarios.

## Table of Contents

1. [Recipe 1: Retail Daily Sales](#recipe-1-retail-daily-sales)
2. [Recipe 2: Industrial Hourly Metrics](#recipe-2-industrial-hourly-metrics)
3. [Recipe 3: Intermittent Demand Forecasting](#recipe-3-intermittent-demand-forecasting)
4. [Recipe 4: Hierarchical Forecasting](#recipe-4-hierarchical-forecasting)
5. [Recipe 5: TSFM Model Selection](#recipe-5-tsfm-model-selection)
6. [Recipe 6: Custom Model Integration](#recipe-6-custom-model-integration)
7. [Recipe 7: Backtest Analysis](#recipe-7-backtest-analysis)
8. [Recipe 8: Error Handling](#recipe-8-error-handling)

---

## Recipe 1: Retail Daily Sales

**Scenario**: Forecast daily sales for multiple retail stores.

**What you'll learn**:
- Basic forecasting workflow
- Using quantiles for uncertainty
- Quick vs standard mode

```python
import pandas as pd
from tsagentkit import TaskSpec, run_forecast

# Create sample retail data
dates = pd.date_range("2024-01-01", periods=90, freq="D")
np.random.seed(42)

# Store A: Strong weekly pattern
store_a = pd.DataFrame({
    "unique_id": "store_a",
    "ds": dates,
    "y": 100 + 20 * np.sin(np.arange(90) * 2 * np.pi / 7) + np.random.normal(0, 10, 90),
})

# Store B: Growing trend
store_b = pd.DataFrame({
    "unique_id": "store_b",
    "ds": dates,
    "y": 80 + 0.5 * np.arange(90) + np.random.normal(0, 8, 90),
})

# Store C: Steady with promotions
store_c = pd.DataFrame({
    "unique_id": "store_c",
    "ds": dates,
    "y": 120 + np.random.normal(0, 15, 90),
})
# Add promotion spikes
store_c.loc[::14, "y"] += 50  # Bi-weekly promotions

df = pd.concat([store_a, store_b, store_c], ignore_index=True)

# Define forecasting task
spec = TaskSpec(
    horizon=14,           # Forecast 2 weeks ahead
    freq="D",             # Daily frequency
    quantiles=[0.1, 0.5, 0.9],  # 80% prediction interval
    season_length=7,      # Weekly seasonality
)

# Quick mode: Skip backtest, faster for prototyping
result = run_forecast(df, spec, mode="quick")

# Access forecast
forecast_df = result.forecast.df
print(f"Forecast rows: {len(forecast_df)}")
print(forecast_df.head(10))

# Access provenance
print(f"\nData signature: {result.provenance.data_signature}")
print(f"Run ID: {result.provenance.run_id}")
print(f"Model used: {result.forecast.model_name}")

# Access quantile columns
quantile_cols = result.forecast.get_quantile_columns()
print(f"\nQuantile columns: {quantile_cols}")
```

**Output**:
```
Forecast rows: 42
  unique_id         ds       yhat  q0.10  q0.50  q0.90
0   store_a 2024-04-01  115.2345  98.12 115.23 132.34
1   store_a 2024-04-02  108.4567  91.34 108.46 125.56
...

Data signature: a1b2c3d4e5f67890
Run ID: 20240401_120000_abcdef12
Model used: SeasonalNaive
```

---

## Recipe 2: Industrial Hourly Metrics

**Scenario**: Forecast equipment sensor readings for maintenance planning.

**What you'll learn**:
- Standard mode with backtesting
- Analyzing backtest metrics
- Handling gaps in data

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast
from tsagentkit.series import TSDataset

# Generate hourly sensor data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=30*24, freq="H")  # 30 days

# Temperature sensor with daily cycle
temp = 20 + 5 * np.sin(np.arange(30*24) * 2 * np.pi / 24) + np.random.normal(0, 1, 30*24)

# Pressure sensor with weekly pattern
pressure = 100 + 10 * np.sin(np.arange(30*24) * 2 * np.pi / (24*7)) + np.random.normal(0, 2, 30*24)

df = pd.DataFrame({
    "unique_id": ["temp_sensor"] * len(dates) + ["pressure_sensor"] * len(dates),
    "ds": list(dates) * 2,
    "y": list(temp) + list(pressure),
})

# Introduce some gaps (sensor offline)
df = df.sample(frac=0.95).sort_values(["unique_id", "ds"]).reset_index(drop=True)

# Define task
spec = TaskSpec(
    horizon=24,           # 1 day ahead
    freq="H",             # Hourly
    season_length=24,     # Daily seasonality
)

# Standard mode: Full pipeline with backtest
result = run_forecast(df, spec, mode="standard")

# Analyze results
print(result.summary())

# Access backtest metrics
if result.backtest_report:
    metrics = result.backtest_report.get("aggregate_metrics", {})
    print(f"\nBacktest Metrics:")
    print(f"  WAPE:  {metrics.get('wape', 0):.2%}")
    print(f"  SMAPE: {metrics.get('smape', 0):.2%}")
    print(f"  MASE:  {metrics.get('mase', 0):.2f}")
    print(f"  MAE:   {metrics.get('mae', 0):.2f}")
    print(f"  RMSE:  {metrics.get('rmse', 0):.2f}")

# Check sparsity detection
dataset = TSDataset.from_dataframe(df, spec)
if dataset.sparsity_profile:
    print(f"\nSparsity Profile:")
    for uid in dataset.series_ids:
        classification = dataset.sparsity_profile.get_classification(uid)
        print(f"  {uid}: {classification.value}")
```

---

## Recipe 3: Intermittent Demand Forecasting

**Scenario**: Forecast demand for slow-moving spare parts with many zero values.

**What you'll learn**:
- Handling intermittent demand
- Sparsity detection
- Fallback ladder behavior

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast, make_plan
from tsagentkit.series import TSDataset

# Generate intermittent demand data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=180, freq="D")

# Part A: Very intermittent (70% zeros)
part_a = []
for d in dates:
    if np.random.random() < 0.7:
        part_a.append(0)
    else:
        part_a.append(np.random.poisson(5))

# Part B: Moderately intermittent (40% zeros)
part_b = []
for d in dates:
    if np.random.random() < 0.4:
        part_b.append(0)
    else:
        part_b.append(np.random.poisson(8))

# Part C: Regular demand (10% zeros)
part_c = [max(0, np.random.poisson(12)) for _ in dates]

df = pd.DataFrame({
    "unique_id": ["part_a"] * len(dates) + ["part_b"] * len(dates) + ["part_c"] * len(dates),
    "ds": list(dates) * 3,
    "y": part_a + part_b + part_c,
})

spec = TaskSpec(horizon=30, freq="D")

# Create dataset and check sparsity
dataset = TSDataset.from_dataframe(df, spec)
print("Sparsity Analysis:")
for uid in dataset.series_ids:
    series_data = df[df["unique_id"] == uid]["y"]
    zero_ratio = (series_data == 0).mean()
    classification = dataset.sparsity_profile.get_classification(uid)
    print(f"  {uid}: {zero_ratio:.1%} zeros -> {classification.value}")

# Create plan (router detects intermittent automatically)
plan = make_plan(dataset, spec)
print(f"\nPlan: {plan.to_signature()}")
print(f"Strategy: {plan.strategy}")
print(f"Config: {plan.config}")

# Run forecast
result = run_forecast(df, spec)

# Check which model was used
print(f"\nModel used: {result.forecast.model_name}")

# View forecast for intermittent part
part_a_forecast = result.forecast.get_series("part_a")
print(f"\nPart A forecast (first 10 days):")
print(part_a_forecast[["ds", "yhat"]].head(10))
```

---

## Recipe 4: Hierarchical Forecasting

**Scenario**: Forecast sales at multiple aggregation levels (Total -> Category -> Product).

**What you'll learn**:
- Defining hierarchy structures
- Forecast reconciliation
- Coherence checking

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast
from tsagentkit.series import TSDataset
from tsagentkit.hierarchy import (
    HierarchyStructure,
    ReconciliationMethod,
    reconcile_forecasts,
    HierarchyEvaluator,
)

# Create hierarchical data
dates = pd.date_range("2024-01-01", periods=60, freq="D")
np.random.seed(42)

# Bottom level: Products
laptops = pd.DataFrame({
    "unique_id": "laptops",
    "ds": dates,
    "y": 50 + np.random.poisson(10, len(dates)),
})

phones = pd.DataFrame({
    "unique_id": "phones",
    "ds": dates,
    "y": 80 + np.random.poisson(15, len(dates)),
})

shirts = pd.DataFrame({
    "unique_id": "shirts",
    "ds": dates,
    "y": 30 + np.random.poisson(8, len(dates)),
})

pants = pd.DataFrame({
    "unique_id": "pants",
    "ds": dates,
    "y": 25 + np.random.poisson(6, len(dates)),
})

# Aggregate to categories
electronics = pd.DataFrame({
    "unique_id": "electronics",
    "ds": dates,
    "y": laptops["y"].values + phones["y"].values,
})

clothing = pd.DataFrame({
    "unique_id": "clothing",
    "ds": dates,
    "y": shirts["y"].values + pants["y"].values,
})

# Aggregate to total
total = pd.DataFrame({
    "unique_id": "total",
    "ds": dates,
    "y": electronics["y"].values + clothing["y"].values,
})

# Combine all levels
df = pd.concat([laptops, phones, shirts, pants, electronics, clothing, total], ignore_index=True)

# Define hierarchy structure
hierarchy = HierarchyStructure(
    aggregation_graph={
        "total": ["electronics", "clothing"],
        "electronics": ["laptops", "phones"],
        "clothing": ["shirts", "pants"],
    },
    bottom_nodes=["laptops", "phones", "shirts", "pants"],
)

spec = TaskSpec(horizon=14, freq="D")

# Run forecast with hierarchy
result = run_forecast(df, spec, hierarchy=hierarchy)

# Access forecasts at all levels
forecast_df = result.forecast.df
print("Forecast at all hierarchy levels:")
for uid in forecast_df["unique_id"].unique():
    total_forecast = forecast_df[forecast_df["unique_id"] == uid]["yhat"].sum()
    print(f"  {uid}: total forecast = {total_forecast:.1f}")

# Verify coherence manually
total_sum = forecast_df[forecast_df["unique_id"] == "total"]["yhat"].sum()
elec_sum = forecast_df[forecast_df["unique_id"] == "electronics"]["yhat"].sum()
cloth_sum = forecast_df[forecast_df["unique_id"] == "clothing"]["yhat"].sum()

print(f"\nCoherence Check:")
print(f"  Total: {total_sum:.1f}")
print(f"  Electronics + Clothing: {elec_sum + cloth_sum:.1f}")
print(f"  Coherent: {abs(total_sum - (elec_sum + cloth_sum)) < 0.1}")

# Check plan includes hierarchical config
print(f"\nPlan config:")
print(f"  Hierarchical: {result.plan.get('hierarchical', False)}")
print(f"  Reconciliation method: {result.plan.get('reconciliation_method', 'N/A')}")
```

---

## Recipe 5: TSFM Model Selection

**Scenario**: Choose and use Time-Series Foundation Models.

**What you'll learn**:
- TSFM adapter selection
- Checking availability
- Fallback configuration

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast
from tsagentkit.series import TSDataset
from tsagentkit.router import make_plan
from tsagentkit.models.adapters import AdapterRegistry

# Create sample data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="D")
df = pd.DataFrame({
    "unique_id": ["series_a"] * len(dates) + ["series_b"] * len(dates),
    "ds": list(dates) * 2,
    "y": list(np.cumsum(np.random.randn(len(dates))) + 100) * 2,
})

spec = TaskSpec(horizon=14, freq="D")
dataset = TSDataset.from_dataframe(df, spec)

# Check TSFM availability
print("TSFM Availability:")
for name in ["chronos", "moirai", "timesfm"]:
    is_avail, msg = AdapterRegistry.check_availability(name)
    status = "✓ Available" if is_avail else f"✗ {msg}"
    print(f"  {name}: {status}")

# Create plan with TSFM preference
plan = make_plan(
    dataset,
    spec,
    strategy="tsfm_first",
    tsfm_preference=["chronos", "moirai", "timesfm"],
)

print(f"\nPlan created:")
print(f"  Primary: {plan.primary_model}")
print(f"  Fallback chain: {plan.fallback_chain[:5]}...")  # First 5

# Run forecast (will use TSFM if available, else fall back)
result = run_forecast(df, spec)

print(f"\nActual model used: {result.forecast.model_name}")

# View fallback events if any
if result.provenance and result.provenance.fallbacks_triggered:
    print(f"\nFallbacks triggered:")
    for fb in result.provenance.fallbacks_triggered:
        print(f"  {fb['from']} -> {fb['to']}: {fb['error']}")
else:
    print(f"\nNo fallbacks triggered (primary model succeeded)")

# Show final forecast
print(f"\nForecast sample:")
print(result.forecast.df.head())
```

---

## Recipe 6: Custom Model Integration

**Scenario**: Integrate your own forecasting model.

**What you'll learn**:
- Custom fit functions
- Custom predict functions
- Integration with pipeline

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, run_forecast
from tsagentkit.contracts import ModelArtifact, ForecastResult, Provenance
from datetime import datetime, timezone

# Create sample data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=60, freq="D")
df = pd.DataFrame({
    "unique_id": ["A"] * len(dates) + ["B"] * len(dates),
    "ds": list(dates) * 2,
    "y": list(range(len(dates))) + list(range(50, 50 + len(dates))),
})

# Define custom model
class NaiveModel:
    """Simple naive model for demonstration."""
    def __init__(self, season_length=1):
        self.season_length = season_length
        self.last_values = {}

    def fit(self, df):
        """Store last value per series."""
        for uid in df["unique_id"].unique():
            series = df[df["unique_id"] == uid].sort_values("ds")
            self.last_values[uid] = series["y"].iloc[-1]
        return self

    def predict(self, horizon, freq, series_ids):
        """Generate naive forecast."""
        forecasts = []
        last_date = pd.Timestamp.now()

        for uid in series_ids:
            last_val = self.last_values.get(uid, 0)
            for h in range(horizon):
                forecast_date = last_date + pd.Timedelta(days=h+1)
                forecasts.append({
                    "unique_id": uid,
                    "ds": forecast_date,
                    "yhat": last_val,
                })

        return pd.DataFrame(forecasts)

# Custom fit function
def my_fit(dataset, plan):
    """Custom fit function."""
    season_length = plan.config.get("season_length", 1)

    model = NaiveModel(season_length=season_length)
    model.fit(dataset.df)

    return ModelArtifact(
        model=model,
        model_name="NaiveModel",
        config={"season_length": season_length},
        metadata={"custom": True},
    )

# Custom predict function
def my_predict(dataset, artifact, spec):
    """Custom predict function."""
    model = artifact.model

    # Generate dates for forecast
    last_date = dataset.df["ds"].max()
    horizon = spec.horizon

    forecasts = []
    for uid in dataset.series_ids:
        last_val = model.last_values.get(uid, 0)
        for h in range(horizon):
            forecast_date = last_date + pd.Timedelta(days=h+1)
            forecasts.append({
                "unique_id": uid,
                "ds": forecast_date,
                "yhat": last_val,
            })

    forecast_df = pd.DataFrame(forecasts)

    # Create basic provenance
    provenance = Provenance(
        run_id=f"custom_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_signature="custom",
        task_signature=spec.model_hash(),
        plan_signature=artifact.signature,
        model_signature=artifact.signature,
    )

    return ForecastResult(
        df=forecast_df,
        provenance=provenance,
        model_name="NaiveModel",
        horizon=horizon,
    )

# Run with custom functions
spec = TaskSpec(horizon=7, freq="D", season_length=1)
result = run_forecast(
    df,
    spec,
    mode="quick",
    fit_func=my_fit,
    predict_func=my_predict,
)

print(f"Custom model forecast:")
print(result.forecast.df)
print(f"\nModel name: {result.forecast.model_name}")
```

---

## Recipe 7: Backtest Analysis

**Scenario**: Evaluate model performance with rolling window backtesting.

**What you'll learn**:
- Rolling backtest configuration
- Analyzing window results
- Per-series diagnostics

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, make_plan
from tsagentkit.series import TSDataset
from tsagentkit.backtest import rolling_backtest
from tsagentkit.models import fit, predict

# Create sample data with trend
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=200, freq="D")
trend = np.linspace(100, 150, len(dates))
seasonal = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)

df = pd.DataFrame({
    "unique_id": ["sales"] * len(dates),
    "ds": dates,
    "y": trend + seasonal + np.random.normal(0, 5, len(dates)),
})

spec = TaskSpec(horizon=14, freq="D", season_length=7)
dataset = TSDataset.from_dataframe(df, spec)
plan = make_plan(dataset, spec)

# Run rolling backtest with expanding window
print("Running expanding window backtest...")
expanding_report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit,
    predict_func=predict,
    n_windows=5,
    window_strategy="expanding",
    step_size=14,
)

print(f"\nExpanding Window Results:")
print(f"  Windows completed: {expanding_report.n_windows}")
print(f"  Strategy: {expanding_report.strategy}")
print(f"\nAggregate Metrics:")
for name, value in expanding_report.aggregate_metrics.items():
    print(f"  {name}: {value:.4f}")

# Run sliding window backtest
print("\n\nRunning sliding window backtest...")
sliding_report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit,
    predict_func=predict,
    n_windows=3,
    window_strategy="sliding",
    min_train_size=100,
    step_size=14,
)

print(f"\nSliding Window Results:")
print(f"  Windows completed: {sliding_report.n_windows}")

# Analyze window details
print(f"\nWindow Details (Expanding):")
for window in expanding_report.window_results:
    print(f"  Window {window.window_index}:")
    print(f"    Train: {window.train_start} to {window.train_end}")
    print(f"    Test:  {window.test_start} to {window.test_end}")
    print(f"    Series: {window.num_series}, Observations: {window.num_observations}")

# Check for errors
if expanding_report.errors:
    print(f"\nErrors encountered:")
    for error in expanding_report.errors:
        print(f"  Window {error.get('window')}: {error.get('error')}")
else:
    print(f"\nNo errors encountered ✓")

# Print summary
print(f"\n{expanding_report.summary()}")
```

---

## Recipe 8: Error Handling

**Scenario**: Handle common errors and edge cases.

**What you'll learn**:
- Catching specific errors
- Understanding error context
- Recovery strategies

```python
import pandas as pd
import numpy as np
from tsagentkit import (
    TaskSpec, run_forecast, validate_contract,
    EContractMissingColumn, EContractDuplicateKey,
    ESplitRandomForbidden, ECovariateLeakage,
    EFallbackExhausted,
)

# Example 1: Missing columns
print("=== Example 1: Missing Columns ===")
bad_df = pd.DataFrame({
    "id": ["A", "A", "B", "B"],
    "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
    "value": [1.0, 2.0, 3.0, 4.0],
})

try:
    report = validate_contract(bad_df)
    report.raise_if_errors()
except EContractMissingColumn as e:
    print(f"Caught expected error: {e.error_code}")
    print(f"  Message: {e.message}")
    print(f"  Context: {e.context}")
    print(f"  Fix: Rename columns to 'unique_id', 'ds', 'y'")

# Example 2: Duplicate keys
print("\n=== Example 2: Duplicate Keys ===")
dup_df = pd.DataFrame({
    "unique_id": ["A", "A", "A", "B", "B"],
    "ds": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
    "y": [1.0, 1.5, 2.0, 3.0, 4.0],
})

try:
    report = validate_contract(dup_df)
    report.raise_if_errors()
except EContractDuplicateKey as e:
    print(f"Caught expected error: {e.error_code}")
    print(f"  Duplicate (unique_id, ds) pairs found")
    print(f"  Fix: Remove duplicates with df.drop_duplicates(['unique_id', 'ds'])")

# Example 3: Unsorted data (auto-fixed)
print("\n=== Example 3: Unsorted Data ===")
shuffled_df = pd.DataFrame({
    "unique_id": ["B", "A", "B", "A"],
    "ds": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"]),
    "y": [4.0, 1.0, 3.0, 2.0],
})

report = validate_contract(shuffled_df)
if report.errors:
    print(f"Validation error: {report.errors[0]['message']}")
    print(f"Fix: Sort with df.sort_values(['unique_id', 'ds'])")
else:
    print("Data was auto-sorted during validation ✓")

# Example 4: Valid data run
print("\n=== Example 4: Valid Data Run ===")
valid_df = pd.DataFrame({
    "unique_id": ["A"] * 30 + ["B"] * 30,
    "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
    "y": list(range(30)) + list(range(50, 80)),
})

spec = TaskSpec(horizon=7, freq="D")

try:
    result = run_forecast(valid_df, spec, mode="quick")
    print(f"Success! Model: {result.forecast.model_name}")
    print(f"Forecast rows: {len(result.forecast.df)}")
except EFallbackExhausted as e:
    print(f"All models failed: {e.message}")
    print(f"Models attempted: {e.context.get('models_attempted')}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Example 5: View all events in result
print("\n=== Example 5: Event Log ===")
result = run_forecast(valid_df, spec, mode="quick")
events = result.metadata.get("events", [])
print(f"Pipeline events ({len(events)} total):")
for event in events:
    status_icon = "✓" if event.get("status") == "success" else "✗"
    print(f"  {status_icon} {event['step_name']}: {event['status']} ({event['duration_ms']:.1f}ms)")
    if event.get("error_code"):
        print(f"    Error: {event['error_code']}")

print("\n=== All Examples Complete ===")
```

---

## Additional Resources

- **API Reference**: See `tool_map.md` for complete function documentation
- **Getting Started**: See `README.md` for basic concepts
- **Technical Specs**: See `docs/PRD.md` for detailed requirements
- **Troubleshooting**: See `docs/recipes/RECIPE_TROUBLESHOOTING.md` for common issues
