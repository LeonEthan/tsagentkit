# tsagentkit Documentation

Complete documentation for the tsagentkit time-series forecasting engine.

## Getting Started

- [Main README](../README.md) - Overview and quick start
- [Installation](#installation) - Setup instructions
- [Quick Start Guide](#quick-start) - First steps

## TSFM Adapters

Time-Series Foundation Model integration guides:

- [Chronos Adapter](adapters/CHRONOS.md) - Amazon Chronos
- [Moirai Adapter](adapters/MOIRAI.md) - Salesforce Moirai
- [TimesFM Adapter](adapters/TIMESFM.md) - Google TimesFM

### Quick Adapter Selection

| Model | Best For | Multivariate | Speed |
|-------|----------|--------------|-------|
| Chronos | General purpose | No | Fast |
| Moirai | Correlated series | Yes | Fast |
| TimesFM | Long horizons | Limited | Fastest |

## Hierarchical Forecasting

- [Reconciliation Guide](hierarchical/RECONCILIATION.md) - Complete hierarchical forecasting guide

### Supported Methods

1. Bottom-Up (BU)
2. Top-Down (TD)
3. Middle-Out
4. OLS (Ordinary Least Squares)
5. WLS (Weighted Least Squares)
6. MinT (Minimum Trace)

## Agent Recipes

Pre-built recipes for AI agents:

- [TSFM Model Selection](recipes/RECIPE_TSFM_SELECTION.md)
  - Decision trees for model selection
  - Domain-specific recommendations
  - Infrastructure constraints

- [Hierarchical Forecasting](recipes/RECIPE_HIERARCHICAL_FORECASTING.md)
  - Step-by-step hierarchical workflow
  - Common patterns and solutions
  - Integration with pipeline

- [Troubleshooting](recipes/RECIPE_TROUBLESHOOTING.md)
  - Common issues and solutions
  - Diagnostic procedures
  - Performance optimization

## API Reference

### Core Components

```python
from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset
from tsagentkit.router import make_plan
from tsagentkit.serving import run_forecast
```

### Hierarchical

```python
from tsagentkit.hierarchy import (
    HierarchyStructure,
    Reconciler,
    ReconciliationMethod,
    reconcile_forecasts,
)
```

### TSFM Adapters

```python
from tsagentkit.models.adapters import (
    ChronosAdapter,
    MoiraiAdapter,
    TimesFMAdapter,
    AdapterRegistry,
)
```

### Serving

```python
from tsagentkit.serving import (
    TSFMModelCache,
    get_tsfm_model,
    clear_tsfm_cache,
)
```

## Best Practices

### 1. Model Selection

- Start with smaller models for prototyping
- Use `get_tsfm_model()` for caching in production
- Configure fallback ladder for robustness

### 2. Hierarchical Data

- Validate S-matrix before forecasting
- Use MinT for maximum accuracy
- Check coherence scores

### 3. Performance

- Use GPU when available
- Batch process multiple series
- Cache models in serving environment

### 4. Guardrails

- Never disable temporal validation
- Handle `ESplitRandomForbidden` properly
- Monitor for data leakage

## Version History

### v1.0 - Ecosystem (Current)

- TSFM adapters (Chronos, Moirai, TimesFM)
- Hierarchical reconciliation (6 methods)
- Model caching for serving
- Complete documentation and recipes

### v0.2 - Enhanced Robustness

- Drift detection
- Retrain triggers
- Stability metrics

### v0.1 - Minimum Loop

- Core pipeline
- Baseline models
- Backtesting

## Contributing

See [AGENTS.md](../AGENTS.md) for contribution guidelines.

## Support

- GitHub Issues: Bug reports and feature requests
- Recipes: Check `docs/recipes/` for common scenarios
- API Docs: See module docstrings
