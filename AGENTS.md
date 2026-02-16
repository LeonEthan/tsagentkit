# Agent Guidelines

Guidelines for AI agents working with `tsagentkit` — a **nanobot-inspired**, minimalist time-series forecasting toolkit.

## Philosophy

- **Less is More**: ~2,000 lines core code
- **Flat Structure**: Minimal nesting, clear imports
- **Pure Functions**: Functional composition over classes
- **Single Source of Truth**: One registry, one config
- **Research-Ready**: Code that's easy to understand and modify

## Project Structure

```
src/tsagentkit/
├── core/
│   ├── config.py          # Single ForecastConfig (frozen dataclass)
│   ├── dataset.py         # TSDataset wrapper
│   ├── types.py           # Shared types
│   └── errors.py          # 4 error types
│
├── models/
│   ├── registry.py        # SINGLE SOURCE: all model specs
│   ├── protocol.py        # fit() / predict() functions
│   ├── cache.py           # ModelCache: TSFM lifecycle
│   ├── ensemble.py        # ensemble aggregation
│   └── adapters/
│       ├── tsfm/          # Thin TSFM wrappers (~100 lines each)
│       └── baseline/      # Naive, SeasonalNaive
│
├── pipeline.py            # forecast(), run_forecast()
├── inspect.py             # list_models(), check_health()
└── __init__.py            # Clean public API (7 functions)
```

## Commands

```bash
# Setup
uv sync --all-extras

# Test
uv run pytest
uv run pytest --cov=src/tsagentkit

# Type check
uv run mypy src/tsagentkit

# Format
uv run ruff format src/
uv run ruff check src/
```

## Key Principles

### 1. Registry Pattern

**The ONLY place to define models**:

```python
# models/registry.py

REGISTRY: dict[str, ModelSpec] = {
    "chronos": ModelSpec(
        name="chronos",
        adapter_path="tsagentkit.models.adapters.tsfm.chronos",
        config_fields={"context_length": 512},
        requires=["chronos", "torch"],
        is_tsfm=True,
    ),
    # ... add new models here
}
```

### 2. Protocol Over Inheritance

**Functions, not base classes**:

```python
# Good: Pure functions
def fit(spec: ModelSpec, dataset: TSDataset) -> ModelArtifact:
    adapter = _load_adapter(spec.adapter_path)
    return adapter.fit(dataset, **spec.config_fields)

# Bad: Complex inheritance
class BaseModel(ABC):
    @abstractmethod
    def fit(self, ...): ...
```

### 3. Single Config

**One frozen dataclass**:

```python
@dataclass(frozen=True)
class ForecastConfig:
    h: int
    freq: str = "D"
    ensemble_method: Literal["median", "mean"] = "median"
```

### 4. Minimal Error Types

**4 errors cover 99% of cases**:

- `EContract`: Input data invalid
- `ENoTSFM`: No TSFM adapters available
- `EInsufficient`: Not enough TSFMs succeeded
- `ETemporal`: Temporal integrity violation

## Adding a New TSFM (3 Steps)

### Step 1: Create Adapter (~100 lines)

```python
# models/adapters/tsfm/mytsfm.py

_loaded_model = None

def load(context_length: int = 512) -> Any:
    """Load model (called by ModelCache)."""
    global _loaded_model
    if _loaded_model is None:
        from mytsfm import MyTSFMModel
        _loaded_model = MyTSFMModel.from_pretrained("mytsfm/base")
    return _loaded_model

def predict(model: Any, dataset: TSDataset, h: int) -> pd.DataFrame:
    """Generate predictions."""
    predictions = model.predict(dataset.df, h=h)
    return pd.DataFrame({
        "unique_id": dataset.unique_ids,
        "ds": dataset.future_dates(h),
        "yhat": predictions,
    })
```

### Step 2: Register (1 line)

```python
# models/registry.py

REGISTRY = {
    # ... existing ...
    "mytsfm": ModelSpec(
        name="mytsfm",
        adapter_path="tsagentkit.models.adapters.tsfm.mytsfm",
        config_fields={"context_length": 512},
        requires=["mytsfm", "torch"],
        is_tsfm=True,
    ),
}
```

### Step 3: Test

```python
# tests/models/adapters/test_mytsfm.py

def test_mytsfm_basic():
    from tsagentkit.models.registry import REGISTRY
    spec = REGISTRY["mytsfm"]
    artifact = fit(spec, sample_dataset)
    result = predict(spec, artifact, sample_dataset, h=7)
    assert len(result) == 7
```

## Code Style

- **Indentation**: 4 spaces, no tabs
- **Naming**: `snake_case` functions, `PascalCase` classes
- **Types**: Add hints for all public APIs
- **Docstrings**: Clear, with examples

```python
# Good
def ensemble(
    predictions: list[pd.DataFrame],
    method: Literal["median", "mean"] = "median"
) -> pd.DataFrame:
    """Element-wise ensemble of predictions.

    Args:
        predictions: DataFrames with [unique_id, ds, yhat]
        method: Aggregation method

    Returns:
        DataFrame with ensemble predictions
    """
    ...
```

## Testing Guidelines

- **Unit tests**: Individual functions
- **Integration**: Full pipeline
- **Adapters**: Mock + real (`TSFM_RUN_REAL=1`)
- **Property tests**: Ensemble correctness

```python
# Good test name
def test_median_ensemble_with_missing_values():
    ...
```

## Commits & PRs

### Commits
- Clear, imperative: "add chronos adapter"
- Focused, single-purpose

### PRs
- Describe change and scope
- Reference CLAUDE.md design
- Include tests for new adapters

## Quick Reference

### Standard Pipeline (Auto Caching)
```python
from tsagentkit import forecast

# Models loaded once, cached automatically
result = forecast(df, h=7, freq="D")
```

### Batch Processing (Explicit Cache Control)
```python
from tsagentkit import forecast, ModelCache
from tsagentkit.models.registry import REGISTRY

# Preload models for batch processing
models = [m for m in REGISTRY.values() if m.is_tsfm]
ModelCache.preload(models)

# Process many datasets - models stay in memory
for df in dataset_batch:
    result = forecast(df, h=7)

# Free memory when done
ModelCache.unload()
```

### Agent Building
```python
from tsagentkit import validate, TSDataset, fit, predict, ensemble
from tsagentkit.models.registry import REGISTRY
from tsagentkit.models.cache import ModelCache

df = validate(raw_df)
dataset = TSDataset.from_dataframe(df, freq="D")
models = [m for m in REGISTRY.values() if m.is_tsfm]

# Models auto-loaded by cache
artifacts = [ModelCache.get(m) for m in models]
preds = [predict(m, a, dataset, h=7) for m, a in zip(models, artifacts)]
result = ensemble(preds, method="median")
```

### Inspection
```python
from tsagentkit.inspect import list_models, check_health
from tsagentkit.models.cache import ModelCache

print(list_models(tsfm_only=True))
print(ModelCache.list_loaded())  # Check cached models
health = check_health()
```

## References

- `CLAUDE.md`: Core design philosophy
- `skill/README.md`: Module guide
- `skill/tool_map.md`: API reference
- `skill/recipes.md`: Examples
