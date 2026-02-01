# Recipe: TSFM Model Selection

**Purpose**: Guide for selecting the appropriate Time-Series Foundation Model for your forecasting task.

**Target**: AI Agents / LLMs

## Quick Decision Tree

```
Start
│
├─ Need multivariate forecasting? ──YES──> Moirai
│   (correlated series)
│
├─ Very long horizon (> 60 steps)? ──YES──> TimesFM
│
├─ Limited compute/GPU memory? ──YES──> Chronos (small)
│
├─ Need fastest inference? ──YES──> TimesFM
│
└─ Default recommendation ───────────> Chronos (base)
```

## Detailed Selection Guide

### 1. Data Characteristics

#### Multivariate Series
```python
# Use Moirai when series are correlated
df = pd.DataFrame({
    "unique_id": ["product_a", "product_b"],
    "ds": [...],
    "y": [...],
})
# Product sales often correlated - use Moirai
```

**Indicators for Multivariate**:
- Series are related (e.g., same product category)
- Cross-series patterns exist
- Need to capture correlations

#### Long Horizon Forecasting
```python
# TimesFM excels at horizons > 60
spec = TaskSpec(horizon=90, freq="D")  # Use TimesFM
```

**Horizon Guidelines**:
- < 30 steps: Any TSFM
- 30-60 steps: Chronos, Moirai
- > 60 steps: TimesFM recommended

#### High-Frequency Data
```python
# Hourly or sub-hourly data
hourly_spec = TaskSpec(horizon=24, freq="H")
# All TSFMs work well; TimesFM slightly faster
```

### 2. Infrastructure Constraints

#### Limited GPU Memory
```python
# Use smaller models
AdapterRegistry.get_recommendation(
    gpu_memory_gb=4,
    priority="memory"
)  # Returns: Chronos (small)
```

#### CPU-Only Deployment
```python
# Chronos small works well on CPU
adapter = ChronosAdapter(
    pipeline="small",
    device="cpu",
)
```

#### High-Throughput Serving
```python
# Use model caching
from tsagentkit.serving import get_tsfm_model

# Load once, reuse for all requests
adapter = get_tsfm_model("timesfm")
```

### 3. Accuracy vs Speed Trade-offs

| Priority | Recommendation | Rationale |
|----------|---------------|-----------|
| Maximum Accuracy | Moirai (large) | Best for complex patterns |
| Balanced | Chronos (base) | Good accuracy, reasonable speed |
| Maximum Speed | TimesFM | Fastest inference |

### 4. Domain-Specific Recommendations

#### E-commerce / Retail
```python
# Often hierarchical with correlations
plan = make_plan(
    dataset,
    spec,
    use_tsfm=True,
    tsfm_preference=["moirai", "chronos"],  # Try Moirai first for correlations
)
```

#### Financial Markets
```python
# Long horizons, multiple frequencies
spec = TaskSpec(horizon=252, freq="D")  # Trading days
adapter = TimesFMAdapter()  # Best for long horizons
```

#### IoT / Sensor Data
```python
# High frequency, many series
hourly_spec = TaskSpec(horizon=168, freq="H")  # One week
adapter = ChronosAdapter(pipeline="small")  # Fast, efficient
```

#### Energy / Utilities
```python
# Strong seasonality, long history
# Moirai handles long context well
adapter = MoiraiAdapter(
    model_size="base",
    context_length=2048,
)
```

### 5. Fallback Strategy

Always configure fallback ladder:

```python
from tsagentkit.router import make_plan

plan = make_plan(
    dataset,
    spec,
    strategy="tsfm_first",
    tsfm_preference=["moirai", "chronos", "timesfm"],
    # Falls back automatically if TSFMs fail
)
```

### 6. Validation Checklist

Before finalizing TSFM selection:

- [ ] Test on holdout data
- [ ] Verify inference latency meets requirements
- [ ] Confirm GPU memory availability
- [ ] Check model licensing for your use case
- [ ] Validate quantile forecasts if needed

### 7. Common Mistakes

**Don't**:
- Use large models for simple patterns (waste of resources)
- Ignore fallback configuration
- Assume bigger is always better

**Do**:
- Start with smaller models for prototyping
- Benchmark on your specific data
- Use caching in production

## Code Template

```python
from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset
from tsagentkit.router import make_plan
from tsagentkit.serving import get_tsfm_model

# 1. Analyze data characteristics
dataset = TSDataset.from_dataframe(df, spec)

# 2. Get automatic recommendation
plan = make_plan(
    dataset,
    spec,
    strategy="tsfm_first",
    tsfm_preference=["moirai", "chronos", "timesfm"],
)

# 3. Load recommended model (with caching)
primary_model = plan.primary_model
if primary_model.startswith("tsfm-"):
    tsfm_name = primary_model.replace("tsfm-", "")
    adapter = get_tsfm_model(tsfm_name)

# 4. Generate forecast
forecast = adapter.fit_predict(dataset, spec)
```

## References

- See `docs/adapters/CHRONOS.md` for Chronos details
- See `docs/adapters/MOIRAI.md` for Moirai details
- See `docs/adapters/TIMESFM.md` for TimesFM details
