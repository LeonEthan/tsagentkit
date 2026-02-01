# Moirai Adapter

Time-Series Foundation Model adapter for [Salesforce Moirai](https://github.com/SalesforceAIResearch/uni2ts).

## Overview

Moirai is a Universal Time Series Transformer trained on Large-Scale data. It uses a transformer architecture with specialized tokenization for time series data and supports both univariate and multivariate forecasting.

## Installation

```bash
pip install uni2ts
```

Or with uv:

```bash
uv pip install uni2ts
```

## Usage

### Basic Usage

```python
from tsagentkit.models.adapters import MoiraiAdapter

# Initialize adapter
adapter = MoiraiAdapter(
    model_size="small",  # Options: "small", "base", "large"
    device="auto",
)

# Prepare data
import pandas as pd
from tsagentkit import TaskSpec
from tsagentkit.series import TSDataset

df = pd.DataFrame({
    "unique_id": ["A"] * 30,
    "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
    "y": range(30),
})
spec = TaskSpec(horizon=7, freq="D")
dataset = TSDataset.from_dataframe(df, spec)

# Generate forecast
forecast = adapter.fit_predict(dataset, spec)
```

### Multivariate Forecasting

```python
# Moirai supports multivariate forecasting
df = pd.DataFrame({
    "unique_id": ["A", "B"] * 30,
    "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
    "y": range(60),
})
dataset = TSDataset.from_dataframe(df, spec)

# Forecast multiple series simultaneously
forecast = adapter.fit_predict(dataset, spec)
```

### Using Model Cache

```python
from tsagentkit.serving import get_tsfm_model

adapter = get_tsfm_model("moirai", model_size="base")
forecast = adapter.fit_predict(dataset, spec)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | "small" | Model size: "small", "base", "large" |
| `device` | str | "auto" | Device for inference |
| `quantiles` | list | None | Quantiles for probabilistic forecasts |
| `context_length` | int | None | Context window (auto-detected if None) |

## Model Sizes

| Size | Parameters | Context Length | Best For |
|------|------------|----------------|----------|
| `small` | ~14M | Up to 512 | Resource-constrained environments |
| `base` | ~46M | Up to 1024 | Balanced performance |
| `large` | ~154M | Up to 2048 | Maximum accuracy |

## Key Features

1. **Any-Variate**: Handles both univariate and multivariate series
2. **Any-Context**: Adapts to different context lengths
3. **Any-Horizon**: Flexible prediction horizons
4. **Probabilistic**: Built-in quantile prediction

## Hardware Requirements

| Size | Minimum RAM | Recommended VRAM |
|------|-------------|------------------|
| `small` | 4GB | 2GB |
| `base` | 8GB | 4GB |
| `large` | 16GB | 10GB |

## Best Practices

1. **Context Length**: Let Moirai auto-detect context length for optimal performance
2. **Multivariate**: Use for datasets with correlated series
3. **Caching**: Always cache models in production serving
4. **Batch Processing**: Group similar-length series for efficiency

## Comparison with Other TSFMs

| Feature | Moirai | Chronos | TimesFM |
|---------|--------|---------|---------|
| Multivariate | Yes | No | Limited |
| Context Flexibility | High | Medium | Medium |
| Speed | Fast | Fast | Fastest |
| Accuracy | High | High | High |

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce context length
adapter = MoiraiAdapter(
    model_size="large",
    context_length=512,  # Limit context
)
```

### Slow on CPU

- Use `model_size="small"` for CPU inference
- Consider using Chronos or TimesFM for CPU-only deployment

## References

- [Moirai Paper](https://arxiv.org/abs/2402.02592)
- [uni2ts GitHub](https://github.com/SalesforceAIResearch/uni2ts)
