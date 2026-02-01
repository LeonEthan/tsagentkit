# tsagentkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust execution engine for AI agents performing time-series forecasting. Provides strict guardrails to enforce proper time-series practices while offering TSFM-first (Time-Series Foundation Model) forecasting with automatic fallback to statistical models.

## Features

- **TSFM-First Strategy**: Priority to Chronos, Moirai, and TimesFM with automatic fallback ladder
- **Hierarchical Reconciliation**: Full support for hierarchical time series with 6 reconciliation methods
- **Strict Guardrails**: Prevents data leakage, enforces temporal integrity, bans random splits
- **Model Caching**: Efficient TSFM model caching for production serving
- **Provenance Tracking**: Complete audit trail with signatures for reproducibility
- **Monitoring**: Drift detection and automated retrain triggers

## Installation

```bash
pip install tsagentkit
```

Or with uv:

```bash
uv pip install tsagentkit
```

### Optional Dependencies

For TSFM support, install the relevant packages:

```bash
# Chronos (Amazon)
pip install chronos-forecasting

# Moirai (Salesforce)
pip install uni2ts

# TimesFM (Google)
pip install timesfm
```

## Quick Start

### Basic Forecasting

```python
import pandas as pd
from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset
from tsagentkit.serving import run_forecast

# Prepare data
df = pd.DataFrame({
    "unique_id": ["A"] * 30,
    "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
    "y": range(30),
})

# Create task spec
spec = TaskSpec(horizon=7, freq="D")

# Run forecast (uses best available model)
artifact = run_forecast(df, spec, mode="standard")
print(artifact.forecast)
```

### Hierarchical Forecasting

```python
import numpy as np
from tsagentkit.hierarchy import (
    HierarchyStructure,
    ReconciliationMethod,
    reconcile_forecasts,
)

# Define hierarchy
hierarchy = HierarchyStructure(
    aggregation_graph={"Total": ["A", "B"]},
    bottom_nodes=["A", "B"],
    s_matrix=np.array([
        [1, 0],  # A
        [0, 1],  # B
        [1, 1],  # Total = A + B
    ]),
)

# Attach to dataset
dataset = dataset.with_hierarchy(hierarchy)

# Forecast with automatic reconciliation
plan = make_plan(dataset, spec)
```

### Using TSFM Models

```python
from tsagentkit.serving import get_tsfm_model

# Load cached TSFM model
adapter = get_tsfm_model("chronos", pipeline="base")

# Generate forecast
forecast = adapter.fit_predict(dataset, spec)
```

## Architecture

```
Input Data
    │
    ▼
┌─────────────┐
│  Validation │── Guardrails (E_SPLIT_RANDOM_FORBIDDEN)
└─────────────┘
    │
    ▼
┌─────────────┐
│     QA      │── Leakage detection, data quality
└─────────────┘
    │
    ▼
┌─────────────┐
│   Router    │── TSFM selection, fallback ladder
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│  Backtest   │────▶│  reconcile  │ (if hierarchical)
└─────────────┘     └─────────────┘
    │
    ▼
┌─────────────┐
│    Fit      │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│   Predict   │────▶│  reconcile  │ (if hierarchical)
└─────────────┘     └─────────────┘
    │
    ▼
┌─────────────┐
│   Package   │── Provenance, signatures
└─────────────┘
```

## TSFM Adapters

tsagentkit provides unified adapters for major Time-Series Foundation Models:

### Chronos (Amazon)
```python
from tsagentkit.models.adapters import ChronosAdapter

adapter = ChronosAdapter(pipeline="base", device="auto")
```

### Moirai (Salesforce)
```python
from tsagentkit.models.adapters import MoiraiAdapter

adapter = MoiraiAdapter(model_size="base", device="auto")
```

### TimesFM (Google)
```python
from tsagentkit.models.adapters import TimesFMAdapter

adapter = TimesFMAdapter(checkpoint_path="google/timesfm-1.0-200m")
```

See [docs/adapters/](docs/adapters/) for detailed configuration.

## Hierarchical Reconciliation

6 reconciliation methods available:

| Method | Description | Best For |
|--------|-------------|----------|
| Bottom-Up | Aggregate bottom forecasts | Bottom patterns important |
| Top-Down | Distribute top forecast | Top patterns reliable |
| Middle-Out | Forecast at middle level | Deep hierarchies |
| OLS | Least squares optimal | Balanced approach |
| WLS | Weighted least squares | Different variances |
| MinT | Minimum trace (optimal) | Maximum accuracy |

```python
from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

reconciled = reconcile_forecasts(
    base_forecasts=forecasts,
    structure=hierarchy,
    method=ReconciliationMethod.MIN_TRACE,
)
```

See [docs/hierarchical/RECONCILIATION.md](docs/hierarchical/RECONCILIATION.md) for full guide.

## Guardrails

tsagentkit enforces strict time-series best practices:

- **E_SPLIT_RANDOM_FORBIDDEN**: Random train/test splits are banned
- **E_COVARIATE_LEAKAGE**: Future covariate leakage detection
- **Temporal Integrity**: Data must be temporally ordered

```python
from tsagentkit.contracts import ESplitRandomForbidden

try:
    dataset = TSDataset.from_dataframe(shuffled_df, spec)
except ESplitRandomForbidden as e:
    print(f"Guardrail triggered: {e}")
```

## Agent Recipes

Pre-built recipes for AI agents:

- [TSFM Model Selection](docs/recipes/RECIPE_TSFM_SELECTION.md)
- [Hierarchical Forecasting](docs/recipes/RECIPE_HIERARCHICAL_FORECASTING.md)
- [Troubleshooting](docs/recipes/RECIPE_TROUBLESHOOTING.md)

## Development

### Setup

```bash
git clone https://github.com/yourusername/tsagentkit.git
cd tsagentkit
uv sync
```

### Testing

```bash
uv run pytest
```

### Project Structure

```
tsagentkit/
├── src/tsagentkit/
│   ├── contracts/      # Validation and task specs
│   ├── qa/            # Data quality checks
│   ├── series/        # TSDataset and operations
│   ├── router/        # Model selection and fallback
│   ├── models/        # Model adapters
│   │   └── adapters/  # TSFM adapters (Chronos, Moirai, TimesFM)
│   ├── hierarchy/     # Hierarchical reconciliation
│   ├── backtest/      # Rolling window backtesting
│   ├── serving/       # Inference orchestration
│   └── monitoring/    # Drift detection
├── docs/              # Documentation
│   ├── adapters/      # TSFM adapter guides
│   ├── hierarchical/  # Reconciliation guide
│   └── recipes/       # Agent recipes
└── tests/             # Test suite
```

## Roadmap

- **v0.1** ✅: Minimum loop (contracts, QA, series, router, baselines, backtest)
- **v0.2** ✅: Enhanced robustness (monitoring, drift detection, triggers)
- **v1.0** ✅: Ecosystem (TSFM adapters, hierarchical reconciliation)
- **v1.1**: External model registry, custom adapter API
- **v1.2**: Distributed forecasting, streaming support

## Contributing

Contributions welcome! Please read [AGENTS.md](AGENTS.md) for coding guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Chronos: Amazon Science
- Moirai: Salesforce AI Research
- TimesFM: Google Research
