# Hierarchical Forecast Reconciliation

Comprehensive guide to hierarchical time series forecasting and reconciliation in tsagentkit.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Hierarchy Structure](#hierarchy-structure)
- [Reconciliation Methods](#reconciliation-methods)
- [Integration with Pipeline](#integration-with-pipeline)
- [Evaluation](#evaluation)
- [Best Practices](#best-practices)

## Overview

Hierarchical time series occur when data has a natural aggregation structure:

```
Total Sales
├── Region: North
│   ├── Store: NYC
│   └── Store: Boston
└── Region: South
    ├── Store: Atlanta
    └── Store: Miami
```

**The Coherence Problem**: Individual forecasts may not sum correctly (e.g., Store forecasts ≠ Region forecast).

**Reconciliation** ensures forecasts respect the hierarchy constraints.

## Quick Start

```python
import numpy as np
import pandas as pd
from tsagentkit import TaskSpec
from tsagentkit.hierarchy import (
    HierarchyStructure,
    ReconciliationMethod,
    reconcile_forecasts,
)
from tsagentkit.series import TSDataset

# 1. Create hierarchical data
df = pd.DataFrame({
    "unique_id": ["A", "A", "B", "B", "Total", "Total"],
    "ds": pd.to_datetime(["2024-01-01", "2024-01-02"] * 3),
    "y": [10.0, 12.0, 20.0, 22.0, 30.0, 34.0],
})

spec = TaskSpec(horizon=7, freq="D")
dataset = TSDataset.from_dataframe(df, spec)

# 2. Define hierarchy
hierarchy = HierarchyStructure(
    aggregation_graph={"Total": ["A", "B"]},
    bottom_nodes=["A", "B"],
    s_matrix=np.array([
        [1, 0],  # A
        [0, 1],  # B
        [1, 1],  # Total = A + B
    ]),
)

# 3. Attach to dataset
dataset = dataset.with_hierarchy(hierarchy)

# 4. Generate base forecasts (using any model)
# ... your forecasting code here ...

# 5. Reconcile forecasts
reconciled = reconcile_forecasts(
    base_forecasts=forecast_df,
    structure=hierarchy,
    method=ReconciliationMethod.MIN_TRACE,
)
```

## Hierarchy Structure

### Defining Hierarchies

#### From Aggregation Graph

```python
from tsagentkit.hierarchy import HierarchyStructure

hierarchy = HierarchyStructure(
    aggregation_graph={
        "Total": ["North", "South"],
        "North": ["NYC", "Boston"],
        "South": ["Atlanta", "Miami"],
    },
    bottom_nodes=["NYC", "Boston", "Atlanta", "Miami"],
    s_matrix=s_matrix,  # Summation matrix
)
```

#### From DataFrame

```python
# Data with hierarchical columns
sales_df = pd.DataFrame({
    "region": ["North", "North", "South", "South"],
    "store": ["NYC", "Boston", "Atlanta", "Miami"],
    "y": [100, 150, 200, 120],
})

hierarchy = HierarchyStructure.from_dataframe(
    sales_df,
    hierarchy_columns=["region", "store"],
)
```

#### From Summation Matrix

```python
import numpy as np

# Direct S matrix definition
s_matrix = np.array([
    # NYC, Boston, Atlanta, Miami
    [1, 0, 0, 0],      # NYC
    [0, 1, 0, 0],      # Boston
    [0, 0, 1, 0],      # Atlanta
    [0, 0, 0, 1],      # Miami
    [1, 1, 0, 0],      # North
    [0, 0, 1, 1],      # South
    [1, 1, 1, 1],      # Total
])

hierarchy = HierarchyStructure.from_summation_matrix(
    s_matrix=s_matrix,
    node_names=["NYC", "Boston", "Atlanta", "Miami", "North", "South", "Total"],
    bottom_node_names=["NYC", "Boston", "Atlanta", "Miami"],
)
```

### The Summation Matrix (S)

The S matrix encodes aggregation relationships:

```
S[i, j] = 1 if bottom node j contributes to node i
```

Example for simple hierarchy:
```
        A   B
A       1   0
B       0   1
Total   1   1   <- Total = A + B
```

## Reconciliation Methods

### 1. Bottom-Up (BU)

**Concept**: Forecast bottom-level series, aggregate up.

**Best For**: When bottom-level patterns are most important.

```python
from tsagentkit.hierarchy import Reconciler, ReconciliationMethod

reconciler = Reconciler(
    method=ReconciliationMethod.BOTTOM_UP,
    structure=hierarchy,
)
reconciled = reconciler.reconcile(base_forecasts)
```

**Pros**:
- Simple, no information loss
- Respects bottom-level patterns

**Cons**:
- Ignores upper-level patterns
- Can be noisy for high-level aggregates

### 2. Top-Down (TD)

**Concept**: Forecast top level, distribute down using proportions.

**Best For**: When top-level patterns are reliable.

```python
# With historical proportions
reconciler = Reconciler(
    method=ReconciliationMethod.TOP_DOWN,
    structure=hierarchy,
)
reconciled = reconciler.reconcile(
    base_forecasts,
    fitted_values=historical_data,
)
```

**Pros**:
- Stable high-level forecasts
- Good for sparse bottom series

**Cons**:
- Loses bottom-level information
- Proportions may not be stable

### 3. Middle-Out

**Concept**: Forecast at middle level, aggregate up and distribute down.

**Best For**: Deep hierarchies where middle level is most reliable.

```python
reconciler = Reconciler(
    method=ReconciliationMethod.MIDDLE_OUT,
    structure=hierarchy,
)
# Specify middle level
reconciled = reconciler.reconcile(
    base_forecasts,
    middle_level=1,
)
```

### 4. OLS (Ordinary Least Squares)

**Concept**: Optimal combination using least squares.

**Best For**: Balanced approach, unbiased forecasts.

```python
reconciler = Reconciler(
    method=ReconciliationMethod.OLS,
    structure=hierarchy,
)
reconciled = reconciler.reconcile(base_forecasts)
```

### 5. WLS (Weighted Least Squares)

**Concept**: Weighted combination based on forecast reliability.

**Best For**: When series have different variances.

```python
reconciler = Reconciler(
    method=ReconciliationMethod.WLS,
    structure=hierarchy,
)
# Provide weights (inverse variance)
weights = np.array([1.0, 1.0, 0.5])  # Lower weight for less reliable series
reconciled = reconciler.reconcile(
    base_forecasts,
    weights=weights,
)
```

### 6. MinT (Minimum Trace)

**Concept**: Optimal reconciliation minimizing forecast error variance.

**Best For**: Maximum accuracy, when residuals available.

```python
reconciler = Reconciler(
    method=ReconciliationMethod.MIN_TRACE,
    structure=hierarchy,
)
reconciled = reconciler.reconcile(
    base_forecasts,
    residuals=residuals,  # In-sample residuals
)
```

**Pros**:
- Minimum error variance
- Theoretically optimal

**Cons**:
- Requires residual estimation
- More computationally intensive

## Integration with Pipeline

### Automatic Reconciliation

```python
from tsagentkit.router import make_plan
from tsagentkit.backtest import rolling_backtest

# Create plan with hierarchical configuration
plan = make_plan(dataset, spec)
# Automatically detects hierarchy and configures reconciliation

# Backtest with automatic reconciliation
report = rolling_backtest(
    dataset=dataset,
    spec=spec,
    plan=plan,
    fit_func=fit_model,
    predict_func=predict_model,
    reconcile=True,  # Enabled by default for hierarchical data
)
```

### Manual Reconciliation

```python
from tsagentkit.hierarchy import reconcile_forecasts

# Get base forecasts from your model
base_forecasts = model.predict(dataset, horizon)

# Reconcile
reconciled = reconcile_forecasts(
    base_forecasts=base_forecasts,
    structure=dataset.hierarchy,
    method=ReconciliationMethod.MIN_TRACE,
)
```

## Evaluation

### Coherence Checking

```python
from tsagentkit.hierarchy import HierarchyEvaluator

evaluator = HierarchyEvaluator(hierarchy)
report = evaluator.evaluate_coherence(forecasts)

print(f"Coherence score: {report.coherence_score}")
for violation in report.violations:
    print(f"Violation: {violation.node} - {violation.gap}")
```

### Accuracy Metrics

```python
# Compute metrics at each level
level_metrics = {}
for level in range(hierarchy.get_num_levels()):
    level_nodes = hierarchy.get_nodes_at_level(level)
    level_forecasts = forecasts[forecasts["unique_id"].isin(level_nodes)]
    level_metrics[level] = compute_metrics(level_forecasts, actuals)
```

## Best Practices

### 1. Method Selection

| Scenario | Recommended Method |
|----------|-------------------|
| Bottom patterns important | Bottom-Up |
| Top patterns reliable | Top-Down |
| Balanced approach | OLS |
| Maximum accuracy | MinT |
| Deep hierarchy | Middle-Out |

### 2. Hierarchy Design

- **Keep it simple**: Avoid very deep hierarchies (>4 levels)
- **Balance matters**: Similar-sized groups work better
- **Sparse series**: Use Top-Down or WLS for intermittent bottom series

### 3. Performance Tips

```python
# Cache reconciler for repeated use
reconciler = Reconciler(method, hierarchy)

# Reconcile multiple horizons at once
reconciled = reconciler.reconcile(base_forecasts)  # (n_nodes, horizon)

# Use Bottom-Up for fastest reconciliation
reconciler = Reconciler(ReconciliationMethod.BOTTOM_UP, hierarchy)
```

### 4. Common Pitfalls

**Don't**:
- Reconcile after every single series forecast
- Use MinT without sufficient residual history
- Ignore coherence violations in evaluation

**Do**:
- Validate hierarchy structure before forecasting
- Compare reconciled vs base forecast accuracy
- Monitor coherence scores in production

## Advanced Topics

### Custom Reconciliation

```python
from tsagentkit.hierarchy.aggregation import create_custom_matrix

# Create custom projection matrix
p_matrix = create_custom_matrix(hierarchy, custom_weights)

# Apply reconciliation
reconciled = hierarchy.s_matrix @ p_matrix @ base_forecasts
```

### Cross-Temporal Reconciliation

For reconciling across both hierarchy and temporal dimensions:

```python
# First temporal, then hierarchical
temporal_reconciled = temporal_reconciliation(forecasts)
final = hierarchical_reconciliation(temporal_reconciled)
```

## References

- [Hyndman et al. (2011) - Forecasting with Temporal Hierarchies](https://robjhyndman.com/papers/temporalhierarchies.pdf)
- [Wickramasuriya et al. (2019) - Optimal Forecast Reconciliation](https://doi.org/10.1016/j.ijforecast.2019.05.020)
- [Athanasopoulos et al. (2020) - Hierarchical Forecasting](https://doi.org/10.1016/j.ijforecast.2019.06.001)
