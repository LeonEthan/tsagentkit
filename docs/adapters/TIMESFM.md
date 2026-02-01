# TimesFM Adapter

Time-Series Foundation Model adapter for [Google TimesFM](https://github.com/google-research/timesfm).

## Overview

TimesFM is a pretrained time-series foundation model from Google Research. It uses a decoder-only transformer architecture optimized for time series forecasting with patch-based tokenization.

## Installation

```bash
pip install timesfm
```

Or with uv:

```bash
uv pip install timesfm
```

## Usage

### Basic Usage

```python
from tsagentkit.models.adapters import TimesFMAdapter

# Initialize adapter
adapter = TimesFMAdapter(
    checkpoint_path="google/timesfm-1.0-200m",
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

### With Custom Configuration

```python
from tsagentkit.models.adapters import AdapterConfig

config = AdapterConfig(
    checkpoint_path="google/timesfm-1.0-200m",
    device="cuda",
    quantiles=[0.1, 0.5, 0.9],
    patch_length=32,
    num_layers=20,
)

adapter = TimesFMAdapter.from_config(config)
```

### Using Model Cache

```python
from tsagentkit.serving import get_tsfm_model

adapter = get_tsfm_model(
    "timesfm",
    checkpoint_path="google/timesfm-1.0-200m"
)
forecast = adapter.fit_predict(dataset, spec)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | str | Required | HuggingFace model identifier |
| `device` | str | "auto" | Device for inference |
| `quantiles` | list | None | Quantiles for probabilistic forecasts |
| `patch_length` | int | 32 | Patch size for tokenization |
| `num_layers` | int | 20 | Number of transformer layers |
| `horizon_len` | int | 128 | Maximum horizon length |

## Available Checkpoints

| Checkpoint | Parameters | Best For |
|------------|------------|----------|
| `google/timesfm-1.0-200m` | 200M | General purpose, highest accuracy |

## Key Features

1. **Patch-based Tokenization**: Efficiently handles long sequences
2. **Decoder-only Architecture**: Fast autoregressive generation
3. **Frequency-aware**: Handles various time frequencies naturally
4. **Zero-shot Capable**: Good performance without fine-tuning

## Hardware Requirements

| Model | Minimum RAM | Recommended VRAM |
|-------|-------------|------------------|
| 200M | 8GB | 6GB |

## Performance Characteristics

| Metric | Performance |
|--------|-------------|
| Inference Speed | Very Fast |
| Cold Start | Medium (model loading) |
| Memory Usage | Medium |
| Long Horizon | Excellent |

## Best Practices

1. **Checkpoint Caching**: Download checkpoint once, cache locally
2. **Batch Processing**: Process multiple series together
3. **Patch Length**: Use default 32 for most cases
4. **Long Horizons**: Excellent for horizons > 30 steps

## Example: Long Horizon Forecasting

```python
# TimesFM excels at long horizons
spec = TaskSpec(horizon=90, freq="D")  # 90-day forecast

adapter = TimesFMAdapter(checkpoint_path="google/timesfm-1.0-200m")
forecast = adapter.fit_predict(dataset, spec)
```

## Example: Multiple Frequencies

```python
# TimesFM handles various frequencies
hourly_spec = TaskSpec(horizon=24, freq="H")
daily_spec = TaskSpec(horizon=30, freq="D")
weekly_spec = TaskSpec(horizon=12, freq="W")

# Same adapter works for all
for spec in [hourly_spec, daily_spec, weekly_spec]:
    forecast = adapter.fit_predict(dataset, spec)
```

## Troubleshooting

### Model Download Issues

```python
# Pre-download checkpoint
from huggingface_hub import snapshot_download

snapshot_download("google/timesfm-1.0-200m")
```

### Memory Issues

```python
# Reduce batch size or use CPU
adapter = TimesFMAdapter(
    checkpoint_path="google/timesfm-1.0-200m",
    device="cpu",
)
```

## Comparison

| Aspect | TimesFM | Chronos | Moirai |
|--------|---------|---------|--------|
| Architecture | Decoder-only | Encoder-decoder | Encoder-decoder |
| Tokenization | Patches | Time tokens | Value tokens |
| Long Horizon | Excellent | Good | Good |
| Speed | Fastest | Fast | Fast |

## References

- [TimesFM Paper](https://arxiv.org/abs/2310.03589)
- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [HuggingFace Model](https://huggingface.co/google/timesfm-1.0-200m)
