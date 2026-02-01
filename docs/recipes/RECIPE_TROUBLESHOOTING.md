# Recipe: Troubleshooting Common Issues

**Purpose**: Diagnose and resolve common issues in time series forecasting pipelines.

**Target**: AI Agents / LLMs

## Issue: Model Fails with "Fallback Exhausted" Error

### Symptoms
```
EFallbackExhausted: All models in the fallback ladder failed
```

### Diagnosis

```python
from tsagentkit.router import FallbackLadder

# Check available models
print(f"Standard ladder: {FallbackLadder.STANDARD_LADDER}")
print(f"Intermittent ladder: {FallbackLadder.INTERMITTENT_LADDER}")

# Check which model is primary
from tsagentkit.router import make_plan
plan = make_plan(dataset, spec)
print(f"Primary: {plan.primary_model}")
print(f"Fallbacks: {plan.fallback_chain}")
```

### Solutions

**1. Check Data Quality**
```python
from tsagentkit.contracts import validate_contract

report = validate_contract(df)
if not report.valid:
    print("Validation errors:", report.errors)
```

**2. Reduce Model Complexity**
```python
# Use simpler baseline models
plan = make_plan(dataset, spec, strategy="baseline_only")
```

**3. Check for Missing Dependencies**
```python
from tsagentkit.models.adapters import AdapterRegistry

# Check TSFM availability
for name in ["chronos", "moirai", "timesfm"]:
    is_avail, msg = AdapterRegistry.check_availability(name)
    print(f"{name}: {is_avail} - {msg}")
```

---

## Issue: TSFM Adapter Not Available

### Symptoms
```
EAdapterNotAvailable: TSFM adapter 'chronos' not found
```

### Diagnosis
```python
# Check if package is installed
try:
    import chronos
    print("Chronos is installed")
except ImportError:
    print("Chronos not installed")
```

### Solutions

```bash
# Install missing packages
pip install chronos-forecasting      # For Chronos
pip install uni2ts                   # For Moirai
pip install timesfm                  # For TimesFM
```

Or use uv:
```bash
uv pip install chronos-forecasting
```

---

## Issue: Out of Memory During TSFM Inference

### Symptoms
```
RuntimeError: CUDA out of memory
```

### Solutions

**1. Use Smaller Model**
```python
# Use small pipeline instead of large
adapter = ChronosAdapter(pipeline="small")  # Instead of "large"
```

**2. Reduce Batch Size**
```python
adapter = ChronosAdapter(
    pipeline="base",
    batch_size=8,  # Reduce from default 32
)
```

**3. Use CPU**
```python
adapter = ChronosAdapter(
    pipeline="base",
    device="cpu",  # Slower but more memory
)
```

**4. Process in Chunks**
```python
# Split dataset into chunks
series_ids = dataset.series_ids
chunk_size = 10

for i in range(0, len(series_ids), chunk_size):
    chunk_ids = series_ids[i:i+chunk_size]
    chunk_dataset = dataset.filter_series(chunk_ids)
    forecast = adapter.fit_predict(chunk_dataset, spec)
```

---

## Issue: Hierarchical Forecasts Not Coherent

### Symptoms
```
Coherence score: 0.85  # Less than 1.0
Violations detected: Total != sum(children)
```

### Diagnosis

```python
from tsagentkit.hierarchy import HierarchyEvaluator

evaluator = HierarchyEvaluator(hierarchy)
report = evaluator.evaluate_coherence(forecasts)

print(f"Coherence score: {report.coherence_score}")
for v in report.violations[:5]:  # Show first 5
    print(f"  {v.node}: gap={v.gap}")
```

### Solutions

**1. Verify S Matrix**
```python
# Check S matrix is correct
print("S matrix shape:", hierarchy.s_matrix.shape)
print("S matrix:")
print(hierarchy.s_matrix)

# Verify: Total row should be all 1s
total_idx = hierarchy.all_nodes.index("Total")
assert all(hierarchy.s_matrix[total_idx] == 1)
```

**2. Use Stronger Reconciliation**
```python
# MinT is more robust than Bottom-Up
from tsagentkit.hierarchy import ReconciliationMethod

reconciled = reconcile_forecasts(
    forecasts,
    hierarchy,
    method=ReconciliationMethod.MIN_TRACE,
    residuals=residuals,  # Provide if available
)
```

**3. Check Data Aggregation**
```python
# Verify historical data sums correctly
for parent, children in hierarchy.aggregation_graph.items():
    parent_data = df[df["unique_id"] == parent]["y"].sum()
    children_sum = df[df["unique_id"].isin(children)]["y"].sum()
    if abs(parent_data - children_sum) > 1e-6:
        print(f"Mismatch: {parent}")
```

---

## Issue: Backtest Fails with "Insufficient Data"

### Symptoms
```
EBacktestInsufficientData: Insufficient data for 5 windows
```

### Diagnosis

```python
# Check data length
print(f"Date range: {dataset.date_range}")
print(f"Unique dates: {dataset.df['ds'].nunique()}")
print(f"Required: {min_train_size + horizon * n_windows}")
```

### Solutions

**1. Reduce Number of Windows**
```python
report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit_func,
    predict_func=predict_func,
    n_windows=3,  # Reduce from 5
)
```

**2. Reduce Horizon**
```python
spec = TaskSpec(horizon=7, freq="D")  # Shorter horizon
```

**3. Reduce Minimum Training Size**
```python
report = rolling_backtest(
    ...,
    min_train_size=20,  # Default might be too high
)
```

---

## Issue: Slow Inference Speed

### Diagnosis

```python
import time

# Benchmark
start = time.time()
forecast = adapter.fit_predict(dataset, spec)
print(f"Time: {time.time() - start:.2f}s")
```

### Solutions

**1. Use Model Caching**
```python
from tsagentkit.serving import get_tsfm_model

# Cache prevents reloading
adapter = get_tsfm_model("chronos", pipeline="base")
```

**2. Use GPU**
```python
adapter = ChronosAdapter(
    pipeline="base",
    device="cuda",  # Instead of "cpu"
)
```

**3. Increase Batch Size**
```python
adapter = ChronosAdapter(
    pipeline="base",
    batch_size=64,  # Increase if memory allows
)
```

**4. Use Faster Model**
```python
# TimesFM is fastest
tsfm_preference=["timesfm", "chronos", "moirai"]
```

---

## Issue: Poor Forecast Accuracy

### Diagnosis

```python
from tsagentkit.backtest import rolling_backtest

# Run backtest to evaluate
report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit_func,
    predict_func=predict_func,
)

print(f"WAPE: {report.get_metric('wape'):.2f}%")
print(f"SMAPE: {report.get_metric('smape'):.2f}%")
print(f"MASE: {report.get_metric('mase'):.2f}")

# Check worst series
worst = report.get_worst_series("wape")
print(f"Worst series: {worst}")
```

### Solutions

**1. Check for Data Issues**
```python
from tsagentkit.qa import run_qa_checks

qa_report = run_qa_checks(dataset)
for issue in qa_report.issues:
    print(f"{issue['severity']}: {issue['type']}")
```

**2. Try Different TSFM**
```python
# Benchmark different models
for model in ["chronos", "moirai", "timesfm"]:
    plan = make_plan(
        dataset, spec,
        tsfm_preference=[model]
    )
    report = rolling_backtest(...)
    print(f"{model}: WAPE={report.get_metric('wape'):.2f}%")
```

**3. Adjust Season Length**
```python
# Explicit season length
spec = TaskSpec(
    horizon=30,
    freq="D",
    season_length=7,  # Weekly seasonality
)
```

**4. Use Fallback Ladder**
```python
# Let router select best model
plan = make_plan(dataset, spec, strategy="auto")
```

---

## Issue: Data Validation Fails

### Symptoms
```
EContractMissingColumn: TSDataset missing required columns: {'ds'}
```

### Solutions

**1. Check Column Names**
```python
print(df.columns)
# Required: unique_id, ds, y

# Rename if needed
df = df.rename(columns={
    "id": "unique_id",
    "date": "ds",
    "value": "y",
})
```

**2. Check Data Types**
```python
print(df.dtypes)

# Fix types
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = pd.to_numeric(df["y"])
```

**3. Check for Duplicates**
```python
duplicates = df.duplicated(["unique_id", "ds"]).sum()
print(f"Duplicates: {duplicates}")

# Remove if needed
df = df.drop_duplicates(["unique_id", "ds"])
```

---

## General Debugging Checklist

```python
def debug_pipeline(df, spec):
    """Run full diagnostic."""
    print("=== Pipeline Debug ===")

    # 1. Validate data
    from tsagentkit.contracts import validate_contract
    report = validate_contract(df)
    print(f"Valid: {report.valid}")
    if not report.valid:
        print(f"Errors: {report.errors}")

    # 2. Check TSFMs
    from tsagentkit.models.adapters import AdapterRegistry
    for name in ["chronos", "moirai", "timesfm"]:
        avail, _ = AdapterRegistry.check_availability(name)
        print(f"{name}: {'✓' if avail else '✗'}")

    # 3. Check hierarchy
    if hasattr(dataset, 'hierarchy') and dataset.hierarchy:
        print(f"Hierarchy: {dataset.hierarchy.get_num_levels()} levels")

    # 4. Memory check
    import psutil
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.available / 1e9:.1f}GB available")

    print("=== End Debug ===")
```

## Getting Help

If issues persist:

1. Check logs in `RunArtifact.metadata["events"]`
2. Review error context: `error.context`
3. Enable verbose logging
4. Consult adapter-specific docs in `docs/adapters/`
