# Recipe: Hierarchical Forecasting

**Purpose**: Step-by-step guide for hierarchical time series forecasting with reconciliation.

**Target**: AI Agents / LLMs

## Scenario

You have sales data with the following hierarchy:
```
Total Sales
├── Electronics
│   ├── Laptops
│   └── Phones
└── Clothing
    ├── Shirts
    └── Pants
```

You need forecasts that respect: `Total = Electronics + Clothing`

## Solution

### Step 1: Prepare Data

```python
import pandas as pd
from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset

# Create hierarchical data
df = pd.DataFrame({
    "category": ["Electronics", "Electronics", "Clothing", "Clothing"] * 30,
    "product": ["Laptops", "Phones", "Shirts", "Pants"] * 30,
    "ds": pd.date_range("2024-01-01", periods=30).tolist() * 4,
    "y": [...],  # Your sales data
})

# Add total level
from tsagentkit.series.dataset import aggregate_to_hierarchy

df_with_total = aggregate_to_hierarchy(
    df,
    hierarchy_columns=["category", "product"],
    value_column="y",
)

spec = TaskSpec(horizon=14, freq="D")
dataset = TSDataset.from_dataframe(df_with_total, spec)
```

### Step 2: Define Hierarchy Structure

```python
from tsagentkit.hierarchy import HierarchyStructure
import numpy as np

# Method 1: From DataFrame columns
hierarchy = HierarchyStructure.from_dataframe(
    df_with_total,
    hierarchy_columns=["category", "product"],
)

# Method 2: Explicit definition
hierarchy = HierarchyStructure(
    aggregation_graph={
        "Total": ["Electronics", "Clothing"],
        "Electronics": ["Laptops", "Phones"],
        "Clothing": ["Shirts", "Pants"],
    },
    bottom_nodes=["Laptops", "Phones", "Shirts", "Pants"],
    s_matrix=np.array([
        # Laptops, Phones, Shirts, Pants
        [1, 0, 0, 0],      # Laptops
        [0, 1, 0, 0],      # Phones
        [0, 0, 1, 0],      # Shirts
        [0, 0, 0, 1],      # Pants
        [1, 1, 0, 0],      # Electronics
        [0, 0, 1, 1],      # Clothing
        [1, 1, 1, 1],      # Total
    ]),
)

# Attach to dataset
dataset = dataset.with_hierarchy(hierarchy)
```

### Step 3: Select Reconciliation Method

```python
from tsagentkit.hierarchy import ReconciliationMethod

# Choose based on your needs:
method = ReconciliationMethod.BOTTOM_UP  # Bottom patterns important
method = ReconciliationMethod.TOP_DOWN   # Top patterns reliable
method = ReconciliationMethod.OLS        # Balanced approach
method = ReconciliationMethod.MIN_TRACE  # Maximum accuracy (recommended)
```

**Decision Guide**:
| If... | Use |
|-------|-----|
| Bottom-level series have clear patterns | Bottom-Up |
| Top-level is stable, bottom is noisy | Top-Down |
| Need balanced approach | OLS |
| Have residuals for training data | MinT |

### Step 4: Generate and Reconcile Forecasts

```python
from tsagentkit.hierarchy import reconcile_forecasts
from tsagentkit.models.adapters import ChronosAdapter

# Generate base forecasts
adapter = ChronosAdapter()
base_forecasts = adapter.fit_predict(dataset, spec)

# Reconcile
reconciled_forecasts = reconcile_forecasts(
    base_forecasts=base_forecasts,
    structure=hierarchy,
    method=ReconciliationMethod.MIN_TRACE,
)

# Verify coherence
from tsagentkit.hierarchy import HierarchyEvaluator

evaluator = HierarchyEvaluator(hierarchy)
report = evaluator.evaluate_coherence(reconciled_forecasts)
print(f"Coherence score: {report.coherence_score:.4f}")
```

### Step 5: Evaluate at Each Level

```python
# Split by hierarchy level for evaluation
for level in range(hierarchy.get_num_levels()):
    level_nodes = hierarchy.get_nodes_at_level(level)
    level_forecasts = reconciled_forecasts[
        reconciled_forecasts["unique_id"].isin(level_nodes)
    ]

    # Compute metrics for this level
    metrics = compute_metrics(level_forecasts, actuals)
    print(f"Level {level}: MAPE = {metrics['mape']:.2f}%")
```

## Complete Pipeline Example

```python
import pandas as pd
import numpy as np
from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset
from tsagentkit.hierarchy import (
    HierarchyStructure,
    ReconciliationMethod,
    reconcile_forecasts,
)
from tsagentkit.models.adapters import ChronosAdapter

# 1. Load and prepare data
df = pd.read_csv("sales_data.csv")
spec = TaskSpec(horizon=14, freq="D")
dataset = TSDataset.from_dataframe(df, spec)

# 2. Build hierarchy from data
hierarchy = HierarchyStructure.from_dataframe(
    df,
    hierarchy_columns=["category", "subcategory", "product"],
)
dataset = dataset.with_hierarchy(hierarchy)

# 3. Generate base forecasts
adapter = ChronosAdapter(pipeline="base")
base_forecasts = adapter.fit_predict(dataset, spec)

# 4. Reconcile
reconciled = reconcile_forecasts(
    base_forecasts,
    hierarchy,
    ReconciliationMethod.MIN_TRACE,
)

# 5. Verify and save
print(f"Hierarchy levels: {hierarchy.get_num_levels()}")
print(f"Total nodes: {len(hierarchy.all_nodes)}")
print(f"Bottom nodes: {len(hierarchy.bottom_nodes)}")

reconciled.to_csv("reconciled_forecasts.csv", index=False)
```

## Common Patterns

### Pattern 1: Deep Hierarchy (>3 levels)

```python
# For deep hierarchies, use Middle-Out
reconciler = Reconciler(
    ReconciliationMethod.MIDDLE_OUT,
    hierarchy,
)
reconciled = reconciler.reconcile(
    base_forecasts,
    middle_level=1,  # Forecast at level 1, reconcile up and down
)
```

### Pattern 2: Intermittent Bottom Series

```python
# For sparse bottom series, use Top-Down
reconciler = Reconciler(
    ReconciliationMethod.TOP_DOWN,
    hierarchy,
)
# Uses historical proportions
total_history = df[df["unique_id"] == "Total"]["y"].values
reconciled = reconciler.reconcile(
    base_forecasts,
    fitted_values=total_history,
)
```

### Pattern 3: Rolling Window with Reconciliation

```python
from tsagentkit.backtest import rolling_backtest

# Automatic reconciliation in backtest
report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit_model,
    predict_func=predict_model,
    reconcile=True,  # Automatically reconciles if hierarchical
)
```

## Troubleshooting

### Issue: Coherence Score is Low

```python
# Check for calculation errors
evaluator = HierarchyEvaluator(hierarchy)
report = evaluator.evaluate_coherence(forecasts)

for violation in report.violations:
    print(f"{violation.node}: expected {violation.expected}, got {violation.actual}")

# Solutions:
# 1. Use MinT instead of Bottom-Up
# 2. Check S matrix is correct
# 3. Verify data aggregation logic
```

### Issue: Bottom-Level Forecasts Too Smooth

```python
# Top-Down may over-smooth bottom series
# Use OLS or MinT to preserve patterns
reconciler = Reconciler(ReconciliationMethod.OLS, hierarchy)
```

### Issue: Memory Issues with Large Hierarchies

```python
# Process in batches
bottom_nodes = hierarchy.bottom_nodes
batch_size = 50

for i in range(0, len(bottom_nodes), batch_size):
    batch_nodes = bottom_nodes[i:i+batch_size]
    batch_forecasts = generate_forecasts(batch_nodes)
    # Reconcile batch
```

## Best Practices

1. **Validate S matrix**: Ensure `Total == sum(bottom)` in historical data
2. **Check coherence**: Always evaluate coherence score
3. **Compare methods**: Benchmark Bottom-Up vs MinT
4. **Monitor levels**: Track accuracy at each hierarchy level
5. **Use residuals**: Provide residuals to MinT for optimal results

## References

- See `docs/hierarchical/RECONCILIATION.md` for full reconciliation guide
- See `docs/recipes/RECIPE_TSFM_SELECTION.md` for model selection
