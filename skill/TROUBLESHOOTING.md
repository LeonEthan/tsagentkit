# tsagentkit Troubleshooting Guide

Error reference with actionable fix hints. All errors inherit from `TSAgentKitError`
and provide `.code`, `.message`, and `.fix_hint` attributes.

---

## Quick Triage

```python
from tsagentkit import forecast
from tsagentkit.core.errors import TSAgentKitError

try:
    result = forecast(df, h=7)
except TSAgentKitError as e:
    print(f"Error: {e.code}")
    print(f"Message: {e.message}")
    print(f"Fix hint: {e.fix_hint}")
```

---

## Error Code Reference

### Contract Errors (`E_CONTRACT`)

| Code | Description | Fix Hint |
|------|-------------|----------|
| `E_CONTRACT` | Input data format invalid | DataFrame must have [unique_id, ds, y] columns |

**Common causes:**
```python
# Missing columns
from tsagentkit import validate
from tsagentkit.core.errors import EContract

try:
    validate(df)
except EContract as e:
    # Column names don't match expected
    # Fix: Rename columns to unique_id, ds, y
    df = df.rename(columns={"id": "unique_id", "date": "ds", "value": "y"})
```

### TSFM Errors

| Code | Description | Fix Hint |
|------|-------------|----------|
| `E_NO_TSFM` | No TSFM models registered | Internal registry error - ensure default TSFM specs exist |
| `E_INSUFFICIENT` | Not enough TSFMs succeeded | Check model compatibility with data frequency and length |

**ENoTSFM causes:**
- TSFM registry is empty (internal error)
- All TSFM adapters failed to import

**EInsufficient causes:**
```python
from tsagentkit import ForecastConfig

# If min_tsfm=2 but only 1 TSFM succeeds
config = ForecastConfig(h=7, min_tsfm=1)  # Reduce requirement
# OR check data compatibility with available models
```

### Temporal Errors (`E_TEMPORAL`)

| Code | Description | Fix Hint |
|------|-------------|----------|
| `E_TEMPORAL` | Temporal integrity violation | Data must be sorted by ds. No future dates in covariates. |

**Common fixes:**
```python
# Ensure data is sorted
df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

# Check for future dates in covariates
max_date = df["ds"].max()
future_cov_dates = future_cov[future_cov["ds"] > max_date]
# These should only be for the forecast horizon
```

---

## Common Issues

### Issue: "Missing required columns"
```python
# Check your column names
print(df.columns)
# Must be exactly: ['unique_id', 'ds', 'y']

# Fix: Rename columns
df = df.rename(columns={
    "series_id": "unique_id",
    "timestamp": "ds",
    "target": "y"
})
```

### Issue: "Column 'y' has null values"
```python
# Remove or fill nulls
df = df.dropna(subset=["unique_id", "ds", "y"])
# OR
df["y"] = df["y"].fillna(method="ffill")  # Forward fill
```

### Issue: Out of Memory with TSFMs
```python
from tsagentkit import ModelCache

# Unload models to free memory
ModelCache.unload()

# For batch processing, process in smaller chunks
```

### Issue: Data too long for model context
```python
from tsagentkit import ForecastConfig
from tsagentkit.models.length_utils import adjust_context_length

# Configure length handling
config = ForecastConfig(
    h=7,
    freq="D",
    context_length=4096,  # Limit context
    strict_length_limits=False,  # Warn instead of error
)
```

### Issue: Model not available
```python
from tsagentkit import check_health

health = check_health()
print(health.tsfm_available)
print(health.tsfm_missing)

# Install missing TSFM packages
# pip install tsagentkit[chronos]
# pip install tsagentkit[timesfm]
# pip install tsagentkit[moirai]
```

---

## Programmatic Error Handling

All errors provide structured access:

```python
from tsagentkit.core.errors import TSAgentKitError, EContract, ENoTSFM, EInsufficient, ETemporal

# Specific error handling
try:
    result = forecast(df, h=7)
except EContract as e:
    # Data format issues
    info = {
        "code": e.code,
        "message": e.message,
        "context": e.context,
        "hint": e.fix_hint,
    }
    print(info)
except (ENoTSFM, EInsufficient) as e:
    # Model availability issues
    print(f"Model issue: {e.hint}")
except ETemporal as e:
    # Sorting/covariate issues
    print(f"Temporal issue: {e.hint}")
```

---

## Related Resources

- **Quickstart**: `skill/QUICKSTART.md` — get running in 3 minutes
- **Recipes**: `skill/recipes.md` — end-to-end templates
- **API Reference**: `skill/tool_map.md` — task-to-API lookup
- **Design**: `docs/DESIGN.md` — system architecture
