# Chronos Adapter

Time-Series Foundation Model adapter for [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting).

## Overview

Chronos is a pretrained time series forecasting model based on language model architectures. It tokenizes time series data and uses a T5-based encoder-decoder architecture for prediction.

## Installation

```bash
pip install chronos-forecasting
```

Or with uv:

```bash
uv pip install chronos-forecasting
```

## Usage

### Basic Usage

```python
from tsagentkit.models.adapters import ChronosAdapter

# Initialize adapter
adapter = ChronosAdapter(
    pipeline="small",  # Options: "small", "base", "large"
    device="auto",     # Auto-detects CUDA/MPS/CPU
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

### With Custom Configuration

```python
from tsagentkit.models.adapters import AdapterConfig

config = AdapterConfig(
    pipeline="large",
    device="cuda",
    quantiles=[0.1, 0.5, 0.9],
)

adapter = ChronosAdapter.from_config(config)
```

### Using Model Cache (Recommended for Serving)

```python
from tsagentkit.serving import get_tsfm_model

# Model is cached and reused
adapter = get_tsfm_model("chronos", pipeline="large")
forecast = adapter.fit_predict(dataset, spec)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | str | "small" | Model size: "small", "base", "large" |
| `device` | str | "auto" | Device for inference: "auto", "cuda", "mps", "cpu" |
| `quantiles` | list | None | Quantiles for probabilistic forecasts |
| `batch_size` | int | 32 | Batch size for prediction |

## Model Sizes

| Pipeline | Parameters | Best For |
|----------|------------|----------|
| `small` | ~8M | Quick prototyping, edge deployment |
| `base` | ~20M | Balanced accuracy/speed |
| `large` | ~46M | Maximum accuracy, server deployment |

## Hardware Requirements

| Pipeline | Minimum RAM | Recommended GPU VRAM |
|----------|-------------|---------------------|
| `small` | 4GB | 2GB |
| `base` | 8GB | 4GB |
| `large` | 16GB | 8GB |

## Fallback Behavior

If Chronos fails during fitting or prediction, tsagentkit's fallback ladder will automatically try:

1. Next available TSFM (Moirai, TimesFM)
2. Lightweight statistical models
3. Baseline models (SeasonalNaive, HistoricAverage)

## Best Practices

1. **Use caching in production**: Always use `get_tsfm_model()` to avoid reloading
2. **Start with "small"**: For development, use small pipeline for faster iteration
3. **Batch predictions**: Process multiple series together for better throughput
4. **GPU recommended**: Large models benefit significantly from GPU acceleration

## Troubleshooting

### Out of Memory

- Use smaller pipeline size
- Reduce batch_size
- Use CPU instead of GPU for large hierarchies

### Slow Inference

- Ensure GPU is being used (`device="cuda"`)
- Increase batch_size if memory allows
- Use model caching

## References

- [Chronos Paper](https://arxiv.org/abs/2403.07815)
- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
