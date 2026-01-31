# tsagentkit v0.2 Development Plan

> **Document Goal**: Technical implementation plan for v0.2 "Enhanced Robustness"
> **Target Audience**: Core developers, AI agents contributing code
> **Status**: Implemented - All phases complete

---

## 1. Overview

### 1.1 Scope

This plan defines the implementation details for v0.2: **Enhanced Robustness**. The goal is to add production-grade monitoring capabilities, advanced routing with data bucketing, and comprehensive feature versioning for full provenance tracking.

### 1.2 Success Criteria

- [x] `monitoring/` module provides drift detection (PSI/KS) with configurable thresholds
- [x] `monitoring/` module tracks prediction stability (jitter, coverage)
- [x] Router supports data bucketing: Head vs Tail series, Short vs Long history
- [x] Feature hashing captures full feature configuration for provenance
- [x] Retrain triggers fire based on drift thresholds or schedules
- [x] All v0.1 tests continue to pass (backward compatibility)
- [x] All 401 tests passing with 93% coverage

---

## 2. Module Implementation Plan

### 2.1 Project Structure

```
tagentkit/
├── src/tsagentkit/
│   ├── ... (v0.1 modules)
│   ├── features/                    # NEW: Feature engineering module
│   │   ├── __init__.py
│   │   ├── factory.py               # FR-10: Feature factory (lags, calendar)
│   │   ├── covariates.py            # FR-11: Known vs observed covariate handling
│   │   ├── versioning.py            # FR-12: Feature config hashing
│   │   └── matrix.py                # FeatureMatrix dataclass
│   ├── monitoring/                  # NEW: Drift detection and monitoring
│   │   ├── __init__.py
│   │   ├── drift.py                 # FR-23: PSI/KS drift detection
│   │   ├── stability.py             # FR-24: Prediction jitter, coverage
│   │   ├── triggers.py              # FR-25: Retrain triggers
│   │   └── report.py                # DriftReport dataclass
│   └── router/
│       ├── ... (v0.1 files)
│       └── bucketing.py             # NEW: Head/Tail, Short/Long bucketing
├── tests/
│   ├── ... (v0.1 tests)
│   ├── features/
│   │   ├── test_factory.py
│   │   ├── test_covariates.py
│   │   └── test_versioning.py
│   ├── monitoring/
│   │   ├── test_drift.py
│   │   ├── test_stability.py
│   │   └── test_triggers.py
│   └── router/test_bucketing.py
└── docs/
    ├── PRD.md
    ├── DEV_PLAN_v0.1.md
    └── DEV_PLAN_v0.2.md (this file)
```

---

## 3. Phase-by-Phase Implementation

### Phase 1: Features Module (Week 1)

**Goal**: Complete feature engineering with full versioning support

#### 3.1.1 New Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
# Add to existing 'models' optional dependencies
models = [
    "statsforecast>=1.7.0",
    "utilsforecast>=0.1.0",
    "scipy>=1.11.0",          # NEW: For PSI/KS tests in monitoring
]

# New optional dependency group for feature engineering
features = [
    "tsfresh>=0.20.0",        # Optional: Automated feature engineering
    "sktime>=0.24.0",         # Optional: Time series transformers
]
```

#### 3.1.2 Features Module (`features/`)

**File: `features/matrix.py`**
- `FeatureMatrix` dataclass to hold engineered features
- Stores: feature DataFrame, config hash, creation timestamp
- Methods: `to_pandas()`, `get_signature()`, `validate()`

```python
@dataclass(frozen=True)
class FeatureMatrix:
    """Container for engineered features with provenance."""

    data: pd.DataFrame
    config_hash: str
    target_col: str = "y"
    feature_cols: list[str] = field(default_factory=list)
    known_covariates: list[str] = field(default_factory=list)
    observed_covariates: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def signature(self) -> str:
        """Return feature matrix signature for provenance."""
        return f"FeatureMatrix(c={self.config_hash},n={len(self.feature_cols)})"
```

**File: `features/factory.py`** (FR-10)
- `FeatureFactory` class for point-in-time safe feature engineering
- Calendar features: day_of_week, month, year, is_holiday
- Lag features: configurable lags with leakage protection
- Rolling statistics: mean, std, min, max with proper window alignment

```python
class FeatureFactory:
    """Point-in-time safe feature engineering."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def create_features(
        self,
        dataset: TSDataset,
        reference_time: datetime | None = None,
    ) -> FeatureMatrix:
        """
        Create features ensuring no lookahead bias.

        Args:
            dataset: Input TSDataset
            reference_time: Cutoff time for point-in-time correctness
                          (defaults to max(ds) - horizon)

        Returns:
            FeatureMatrix with engineered features
        """
        # Implementation must:
        # 1. Filter data <= reference_time
        # 2. Create lags (ensuring lag >= horizon for observed covariates)
        # 3. Create calendar features
        # 4. Create rolling stats with aligned windows
        pass

    def _create_lag_features(self, df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
        """Create lag features with validation."""
        # Lags must be >= horizon for observed covariates to prevent leakage
        pass

    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features."""
        # dayofweek, month, quarter, year, etc.
        pass

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: list[int],
        aggregations: list[str],
    ) -> pd.DataFrame:
        """Create rolling window statistics (right-aligned, no lookahead)."""
        pass
```

**File: `features/covariates.py`** (FR-11)
- `CovariatePolicy` enum: `KNOWN`, `OBSERVED`
- `CovariateManager` class for covariate handling
- Validation that observed covariates don't extend beyond prediction time

```python
class CovariatePolicy(Enum):
    """Policy for handling different covariate types."""

    KNOWN = "known"           # Known for all time steps (e.g., holidays)
    OBSERVED = "observed"     # Observed up to current time only


class CovariateManager:
    """Manage known vs observed covariates with leakage protection."""

    def __init__(
        self,
        known_covariates: list[str],
        observed_covariates: list[str],
    ):
        self.known_covariates = known_covariates
        self.observed_covariates = observed_covariates

    def validate_for_prediction(
        self,
        df: pd.DataFrame,
        forecast_start: datetime,
        horizon: int,
    ) -> None:
        """
        Validate that observed covariates don't leak future information.

        Raises:
            E_COVARIATE_LEAKAGE: If observed covariates extend beyond cutoff
        """
        pass

    def mask_observed_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = "y",
    ) -> pd.DataFrame:
        """
        Mask observed covariates at time t to only use info available at t-1.

        This creates the proper training setup where observed covariates
        are lagged to prevent leakage.
        """
        pass
```

**File: `features/versioning.py`** (FR-12)
- `compute_feature_hash(config: FeatureConfig) -> str`
- Hash includes: feature types, lag specifications, window sizes, aggregations
- Version tracking for feature schema evolution

```python
def compute_feature_hash(config: FeatureConfig) -> str:
    """
    Compute deterministic hash of feature configuration.

    Includes:
    - Lag specifications
    - Calendar feature flags
    - Rolling window configurations
    - Covariate assignments
    """
    config_dict = {
        "lags": sorted(config.lags) if config.lags else [],
        "calendar": sorted(config.calendar_features),
        "rolling": [
            {"window": w, "aggs": sorted(a)}
            for w, a in sorted(config.rolling_windows.items())
        ],
        "known_covariates": sorted(config.known_covariates),
        "observed_covariates": sorted(config.observed_covariates),
    }
    json_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
```

#### Deliverables
- [x] `FeatureMatrix` dataclass with provenance
- [x] `FeatureFactory` with point-in-time safe feature creation
- [x] `CovariateManager` with leakage prevention
- [x] `compute_feature_hash()` for full versioning
- [x] Unit tests for all feature types (79 tests)
- [x] Tests validating no lookahead bias in features

---

### Phase 2: Monitoring Module (Week 1-2)

**Goal**: Production-grade monitoring with drift detection and retrain triggers

#### 3.2.1 Monitoring Module (`monitoring/`)

**File: `monitoring/report.py`**
- `DriftReport` dataclass for drift detection results
- `StabilityReport` dataclass for prediction stability metrics

```python
@dataclass(frozen=True)
class DriftReport:
    """Report from drift detection analysis."""

    drift_detected: bool
    feature_drifts: dict[str, FeatureDriftResult]
    overall_drift_score: float
    threshold_used: float
    reference_timestamp: str
    current_timestamp: str

@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature_name: str
    metric: str  # "psi" or "ks"
    statistic: float
    p_value: float | None  # KS test p-value
    drift_detected: bool
    reference_distribution: dict  # binned histogram or params
    current_distribution: dict
```

**File: `monitoring/drift.py`** (FR-23)
- PSI (Population Stability Index) calculation
- KS (Kolmogorov-Smirnov) test for distribution drift
- Feature-level and dataset-level drift detection

```python
class DriftDetector:
    """Detect data drift between reference and current distributions."""

    def __init__(
        self,
        method: Literal["psi", "ks"] = "psi",
        threshold: float = 0.2,  # PSI: 0.2=warning, 0.25=alert
        n_bins: int = 10,
    ):
        self.method = method
        self.threshold = threshold
        self.n_bins = n_bins

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> DriftReport:
        """
        Detect drift between reference and current datasets.

        Args:
            reference_data: Baseline/reference distribution (training data)
            current_data: Current data to compare (recent observations)
            features: List of features to check (defaults to numeric columns)

        Returns:
            DriftReport with per-feature and overall results
        """
        pass

    def compute_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Population Stability Index.

        PSI interpretation:
        - < 0.1: No significant change
        - 0.1 - 0.2: Moderate change
        - > 0.2: Significant change (drift detected)

        Formula: PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        pass

    def compute_ks_test(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> tuple[float, float]:
        """
        Compute Kolmogorov-Smirnov test statistic and p-value.

        Returns:
            Tuple of (statistic, p_value)
        """
        pass
```

**File: `monitoring/stability.py`** (FR-24)
- Prediction jitter detection (variance in point predictions over time)
- Quantile coverage analysis (are quantiles well-calibrated?)
- Stability metrics for model performance monitoring

```python
class StabilityMonitor:
    """Monitor prediction stability and calibration."""

    def __init__(
        self,
        jitter_threshold: float = 0.1,  # Coefficient of variation threshold
        coverage_tolerance: float = 0.05,  # Allowed deviation from target coverage
    ):
        self.jitter_threshold = jitter_threshold
        self.coverage_tolerance = coverage_tolerance

    def compute_jitter(
        self,
        predictions: list[pd.DataFrame],
        method: Literal["cv", "mad"] = "cv",
    ) -> dict[str, float]:
        """
        Compute prediction jitter across multiple forecast runs.

        Args:
            predictions: List of forecast DataFrames from different runs
            method: "cv" for coefficient of variation, "mad" for median absolute deviation

        Returns:
            Dict mapping unique_id to jitter metric
        """
        pass

    def compute_coverage(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        quantiles: list[float],
    ) -> dict[float, float]:
        """
        Compute empirical coverage for each quantile.

        Args:
            actuals: DataFrame with actual values [unique_id, ds, y]
            forecasts: DataFrame with quantile forecasts [unique_id, ds, q_0.1, ...]
            quantiles: List of quantile levels

        Returns:
            Dict mapping quantile to empirical coverage (0-1)
        """
        pass

    def check_calibration(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        quantiles: list[float],
    ) -> CalibrationReport:
        """
        Check if quantiles are well-calibrated.

        Returns:
            CalibrationReport with coverage metrics and warnings
        """
        pass
```

**File: `monitoring/triggers.py`** (FR-25)
- `RetrainTrigger` class with multiple trigger conditions
- Drift-based triggers (PSI threshold exceeded)
- Schedule-based triggers (time-based retraining)
- Performance-based triggers (metric degradation)

```python
class TriggerType(Enum):
    """Types of retrain triggers."""

    DRIFT = "drift"           # Data drift detected
    SCHEDULE = "schedule"     # Time-based trigger
    PERFORMANCE = "performance"  # Metric degradation
    MANUAL = "manual"         # Explicit manual trigger


@dataclass
class RetrainTrigger:
    """Configuration for retrain triggers."""

    trigger_type: TriggerType
    threshold: float | None = None  # For drift/performance triggers
    metric_name: str | None = None  # For performance triggers
    schedule: str | None = None     # Cron expression for schedule triggers
    enabled: bool = True


class TriggerEvaluator:
    """Evaluate retrain triggers based on monitoring data."""

    def __init__(self, triggers: list[RetrainTrigger]):
        self.triggers = triggers

    def evaluate(
        self,
        drift_report: DriftReport | None = None,
        stability_report: StabilityReport | None = None,
        current_metrics: dict[str, float] | None = None,
        last_train_time: datetime | None = None,
    ) -> list[TriggerResult]:
        """
        Evaluate all triggers and return those that fired.

        Args:
            drift_report: Latest drift detection results
            stability_report: Latest stability metrics
            current_metrics: Current model performance metrics
            last_train_time: When model was last trained

        Returns:
            List of triggered results (empty if no triggers fired)
        """
        pass

    def should_retrain(
        self,
        drift_report: DriftReport | None = None,
        stability_report: StabilityReport | None = None,
        current_metrics: dict[str, float] | None = None,
        last_train_time: datetime | None = None,
    ) -> bool:
        """Return True if any trigger indicates retraining needed."""
        results = self.evaluate(
            drift_report, stability_report, current_metrics, last_train_time
        )
        return any(r.fired for r in results)
```

#### Deliverables
- [x] `DriftDetector` with PSI and KS implementations
- [x] `StabilityMonitor` for jitter and coverage analysis
- [x] `TriggerEvaluator` for retrain decision logic
- [x] Integration with `scipy.stats` for statistical tests
- [x] Unit tests with synthetic drift scenarios (21 tests)
- [x] Tests for trigger evaluation logic (22 tests)

---

### Phase 3: Advanced Router with Bucketing (Week 2)

**Goal**: Implement Head vs Tail and Short vs Long history bucketing

#### 3.3.1 Router Enhancements (`router/bucketing.py`)

**File: `router/bucketing.py`**
- `SeriesBucket` enum: `HEAD`, `TAIL`, `SHORT_HISTORY`, `LONG_HISTORY`
- `DataBucketer` class for bucketing series by characteristics
- Integration with existing router for bucket-specific model selection

```python
class SeriesBucket(Enum):
    """Buckets for series classification."""

    HEAD = "head"                      # High volume/frequent series
    TAIL = "tail"                      # Low volume/infrequent series
    SHORT_HISTORY = "short_history"    # Few observations
    LONG_HISTORY = "long_history"      # Many observations


@dataclass(frozen=True)
class BucketConfig:
    """Configuration for data bucketing thresholds."""

    # Head/Tail thresholds (based on value volume)
    head_quantile_threshold: float = 0.8  # Top 20% by volume = HEAD
    tail_quantile_threshold: float = 0.2  # Bottom 20% by volume = TAIL

    # History length thresholds
    short_history_max_obs: int = 30       # < 30 obs = short history
    long_history_min_obs: int = 365       # > 365 obs = long history

    # Combined classification logic
    prefer_sparsity: bool = True          # Sparsity trumps volume for bucketing


@dataclass(frozen=True)
class BucketProfile:
    """Bucketing profile for a dataset."""

    bucket_assignments: dict[str, set[SeriesBucket]]  # Series -> Buckets
    bucket_stats: dict[SeriesBucket, BucketStatistics]


class DataBucketer:
    """Bucket series by volume and history characteristics."""

    def __init__(self, config: BucketConfig | None = None):
        self.config = config or BucketConfig()

    def bucket_by_volume(
        self,
        df: pd.DataFrame,
        value_col: str = "y",
    ) -> dict[str, SeriesBucket]:
        """
        Bucket series into HEAD/TAIL based on total value volume.

        Args:
            df: DataFrame with [unique_id, y] columns
            value_col: Column containing values to sum

        Returns:
            Dict mapping unique_id to HEAD or TAIL bucket
        """
        pass

    def bucket_by_history_length(
        self,
        df: pd.DataFrame,
    ) -> dict[str, SeriesBucket]:
        """
        Bucket series into SHORT_HISTORY/LONG_HISTORY by observation count.

        Args:
            df: DataFrame with [unique_id] column

        Returns:
            Dict mapping unique_id to SHORT_HISTORY or LONG_HISTORY bucket
        """
        pass

    def create_bucket_profile(
        self,
        dataset: TSDataset,
        sparsity_profile: SparsityProfile | None = None,
    ) -> BucketProfile:
        """
        Create comprehensive bucket profile combining all bucketing strategies.

        A series can belong to multiple buckets (e.g., HEAD + LONG_HISTORY).
        """
        pass

    def get_model_for_bucket(
        self,
        bucket: SeriesBucket,
        sparsity_class: SparsityClass | None = None,
    ) -> str:
        """
        Get recommended model for a given bucket.

        Model recommendations:
        - HEAD + LONG_HISTORY: TSFM (if available) or sophisticated model
        - HEAD + SHORT_HISTORY: Robust local model
        - TAIL: Simple baseline (SeasonalNaive, HistoricAverage)
        - INTERMITTENT: Croston or ADIDA (via statsforecast)
        """
        pass
```

**File: `router/router.py` - Enhanced for v0.2**

Extend existing router with bucket-aware planning:

```python
# Add to existing router.py

def make_bucketed_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    qa: QAReport | None = None,
    bucket_config: BucketConfig | None = None,
) -> dict[SeriesBucket, Plan]:
    """
    Create separate execution plans for each bucket.

    Returns:
        Dict mapping bucket to Plan for that bucket's series
    """
    bucketer = DataBucketer(bucket_config)
    profile = bucketer.create_bucket_profile(dataset, dataset.sparsity_profile)

    plans = {}
    for bucket in SeriesBucket:
        series_in_bucket = [
            uid for uid, buckets in profile.bucket_assignments.items()
            if bucket in buckets
        ]
        if series_in_bucket:
            # Create subset dataset for this bucket
            bucket_dataset = dataset.filter_series(series_in_bucket)
            plans[bucket] = _make_plan_for_bucket(
                bucket_dataset, task_spec, bucket, qa
            )

    return plans


def _make_plan_for_bucket(
    dataset: TSDataset,
    task_spec: TaskSpec,
    bucket: SeriesBucket,
    qa: QAReport | None,
) -> Plan:
    """Create plan optimized for a specific bucket."""
    # Bucket-specific model selection logic
    pass
```

#### Deliverables
- [x] `DataBucketer` with volume and history length bucketing
- [x] `BucketProfile` for comprehensive series classification
- [x] Bucket-aware router with different plans per bucket
- [x] Integration with existing sparsity classification
- [x] Unit tests for bucketing logic (22 tests)
- [x] Tests for bucket-specific model selection

---

### Phase 4: Integration and Testing (Week 3)

**Goal**: Integrate all v0.2 modules and ensure end-to-end functionality

#### 3.4.1 Integration Points

**Serving Integration (`serving/orchestration.py`)**

Extend `run_forecast` to support monitoring:

```python
def run_forecast(
    data: Any,
    task_spec: TaskSpec,
    mode: Literal['quick', 'standard', 'strict'] = 'standard',
    monitoring_config: MonitoringConfig | None = None,
    reference_data: pd.DataFrame | None = None,  # For drift detection
) -> RunArtifact:
    """
    Enhanced run_forecast with optional monitoring.

    If monitoring_config is provided and reference_data is given:
    - Run drift detection on input data vs reference
    - Include drift report in RunArtifact
    - Fire retrain triggers if configured
    """
    pass
```

**Provenance Enhancement (`serving/provenance.py`)**

Add feature signature to provenance:

```python
def create_provenance(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    plan: Plan,
    feature_matrix: FeatureMatrix | None = None,  # NEW: Feature signature
    model_config: dict[str, Any] | None = None,
    qa_repairs: list[dict[str, Any]] | None = None,
    fallbacks_triggered: list[dict[str, Any]] | None = None,
    drift_report: DriftReport | None = None,  # NEW: Drift info
) -> dict[str, Any]:
    """Enhanced provenance with v0.2 fields."""
    provenance = {
        # ... existing fields ...
        "feature_signature": feature_matrix.signature if feature_matrix else None,
        "drift_detected": drift_report.drift_detected if drift_report else None,
    }
    return provenance
```

#### Deliverables
- [x] `MonitoringConfig` dataclass for monitoring configuration
- [x] `run_forecast` extended with monitoring support
- [x] `create_provenance` enhanced with feature and drift info
- [x] Integration tests for all v0.2 modules (14 tests)
- [x] End-to-end workflow tests

---

## 4. Implementation Summary

### v0.2 New Modules

| Module | Files | Key Classes |
|--------|-------|-------------|
| `features/` | 5 files | `FeatureFactory`, `FeatureMatrix`, `FeatureConfig` |
| `monitoring/` | 4 files | `DriftDetector`, `StabilityMonitor`, `TriggerEvaluator` |
| `router/bucketing.py` | 1 file | `DataBucketer`, `BucketProfile`, `BucketConfig` |

### New Dependencies
- `scipy>=1.15.3` - For KS statistical tests

### Test Summary
- **Phase 1 (Features)**: 79 tests
- **Phase 2 (Monitoring)**: 61 tests
- **Phase 3 (Bucketing)**: 22 tests
- **Phase 4 (Integration)**: 14 tests
- **Total**: 415 tests, 93% coverage

### API Additions

```python
from tsagentkit import (
    MonitoringConfig,
    DataBucketer, BucketConfig, SeriesBucket,
)

# Features (submodule)
from tsagentkit.features import (
    FeatureFactory, FeatureConfig, FeatureMatrix,
    CovariateManager, compute_feature_hash,
)

# Monitoring (submodule)
from tsagentkit.monitoring import (
    DriftDetector, StabilityMonitor, TriggerEvaluator,
    RetrainTrigger, TriggerType,
)
```

---

## Appendix: Usage Examples

### Example 1: Feature Engineering with Monitoring

```python
from tsagentkit import TaskSpec, MonitoringConfig, run_forecast
from tsagentkit.features import FeatureFactory, FeatureConfig

# Configure features
feature_config = FeatureConfig(
    lags=[1, 7, 14],
    calendar_features=["dayofweek", "month"],
    rolling_windows={7: ["mean", "std"]},
)

# Configure monitoring
monitoring = MonitoringConfig(
    enabled=True,
    drift_method="psi",
    drift_threshold=0.2,
)

# Run forecast with monitoring
result = run_forecast(
    data=current_data,
    task_spec=TaskSpec(horizon=7, freq="D"),
    monitoring_config=monitoring,
    reference_data=training_data,
)

# Check provenance for drift info
print(result.provenance["drift_detected"])
print(result.provenance["drift_score"])
```

### Example 2: Bucketed Routing

```python
from tsagentkit.router import DataBucketer, BucketConfig

# Create bucketer
config = BucketConfig(
    head_quantile_threshold=0.8,
    short_history_max_obs=50,
)
bucketer = DataBucketer(config)

# Create bucket profile
profile = bucketer.create_bucket_profile(dataset)

# Get model recommendations per bucket
for bucket in SeriesBucket:
    series = profile.get_series_in_bucket(bucket)
    if series:
        model = bucketer.get_model_for_bucket(bucket)
        print(f"{bucket.value}: {len(series)} series -> {model}")
```

---

*Document Version: 0.2.0*
*Last Updated: 2026-01-31*
*Status: Implementation Complete*
- Each new module has corresponding test file
- Mock external dependencies (scipy.stats)
- Test all drift calculation edge cases

**Integration Tests**
- End-to-end with drift detection enabled
- Test bucketed routing on synthetic multi-bucket data
- Test feature factory with known covariates

**Acceptance Test Cases**

| ID | Test | Expected Result |
|----|------|-----------------|
| B1 | Run with PSI drift detection | DriftReport included in RunArtifact |
| B2 | Drift threshold exceeded | Retrain trigger fires |
| B3 | Bucketed routing on mixed dataset | Different models selected per bucket |
| B4 | Feature hash changes with config | Different signatures, provenance updated |
| B5 | Observed covariate validation | E_COVARIATE_LEAKAGE if future values present |
| B6 | Prediction jitter monitoring | StabilityReport with jitter metrics |

---

## 4. Key Technical Decisions

### 4.1 External Dependencies

| Purpose | Package | Rationale |
|---------|---------|-----------|
| Statistical tests | `scipy>=1.11.0` | PSI/KS implementations |
| TS utilities | `utilsforecast` | Validation and preprocessing |
| Feature engineering | `tsfresh` (optional) | Automated feature extraction |
| Time series transforms | `sktime` (optional) | Advanced transformers |

### 4.2 Bucketing Strategy

| Bucket | Characteristics | Recommended Model |
|--------|-----------------|-------------------|
| HEAD + LONG_HISTORY | High volume, rich history | TSFM or AutoML |
| HEAD + SHORT_HISTORY | High volume, limited history | Robust local model (Theta, ETS) |
| TAIL + LONG_HISTORY | Low volume, rich history | Simple baseline + pooling |
| TAIL + SHORT_HISTORY | Low volume, limited history | Global model or simple naive |
| INTERMITTENT | Sporadic demand | Croston, ADIDA, or IMAPA |
| COLD_START | Very few observations | HistoricAverage or global model |

### 4.3 Drift Detection Strategy

| Method | Use Case | Threshold | Notes |
|--------|----------|-----------|-------|
| PSI | Overall distribution drift | 0.2 (warning), 0.25 (alert) | Industry standard |
| KS Test | Statistical significance | p < 0.05 | Requires sufficient sample size |
| Jitter | Prediction stability | CV > 0.1 | Model-specific threshold |
| Coverage | Quantile calibration | ±0.05 from target | For probabilistic forecasts |

---

## 5. API Surface (v0.2 Additions)

### New Public Functions

```python
# features
from tsagentkit.features import (
    FeatureFactory,
    FeatureMatrix,
    CovariateManager,
    compute_feature_hash,
)

# monitoring
from tsagentkit.monitoring import (
    DriftDetector,
    StabilityMonitor,
    TriggerEvaluator,
    DriftReport,
)

# router (enhanced)
from tsagentkit.router import (
    make_plan,
    DataBucketer,          # NEW
    SeriesBucket,          # NEW
    BucketConfig,          # NEW
)
```

### Enhanced Type Hierarchy

```python
# v0.2 extends v0.1 hierarchy
TaskSpec -> Plan -> ModelArtifact -> RunArtifact
                ↓
         BacktestReport
                ↓
         ForecastResult

# v0.2 additions
FeatureConfig -> FeatureMatrix
                     ↓
              DriftReport (via monitoring)
                     ↓
              TriggerResult

BucketConfig -> BucketProfile
                     ↓
              Plan (bucketed)
```

---

## 6. Milestones and Timeline

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | Features Module | FeatureFactory, CovariateManager, versioning |
| 1-2 | Monitoring Module | DriftDetector, StabilityMonitor, triggers |
| 2 | Router Bucketing | DataBucketer, bucket-aware planning |
| 3 | Integration | End-to-end tests, provenance updates, docs |
| 3 | Polish | skill/ docs update, integration tests, v0.2 release |

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| scipy dependency bloat | Low | Optional dependency, lazy import |
| PSI calculation performance | Medium | Use numpy vectorization, cache bins |
| Bucketing threshold tuning | Medium | Configurable thresholds, sensible defaults |
| Feature leakage in complex transforms | High | Strict point-in-time validation, tests |
| API backward compatibility | Medium | All v0.1 APIs remain unchanged |

---

## 8. Post-v0.2 Roadmap

Items deferred to v1.0:
- Hierarchical reconciliation
- External TSFM adapters (Chronos, Moirai, etc.)
- Online learning capabilities
- Distributed/backtesting at scale

---

## Appendix A: Usage Examples

### A1: Drift Detection

```python
from tsagentkit.monitoring import DriftDetector

# Reference data from training
reference_data = load_training_data()

# New incoming data
new_data = load_recent_observations()

detector = DriftDetector(method="psi", threshold=0.2)
report = detector.detect(reference_data, new_data, features=["y", "sales"])

if report.drift_detected:
    print(f"Drift detected! Score: {report.overall_drift_score}")
    for feature, result in report.feature_drifts.items():
        print(f"  {feature}: PSI={result.statistic:.3f}")
```

### A2: Bucketed Routing

```python
from tsagentkit.router import DataBucketer, BucketConfig

config = BucketConfig(
    head_quantile_threshold=0.8,
    short_history_max_obs=50,
)

bucketer = DataBucketer(config)
profile = bucketer.create_bucket_profile(dataset)

print(f"Head series: {len([s for s, b in profile.bucket_assignments.items() if SeriesBucket.HEAD in b])}")
print(f"Tail series: {len([s for s, b in profile.bucket_assignments.items() if SeriesBucket.TAIL in b])}")

# Create bucketed plans
plans = make_bucketed_plan(dataset, task_spec, bucket_config=config)
for bucket, plan in plans.items():
    print(f"{bucket.value}: {plan.primary_model}")
```

### A3: Feature Engineering with Versioning

```python
from tsagentkit.features import FeatureFactory, FeatureConfig

config = FeatureConfig(
    lags=[1, 7, 14],
    calendar_features=["dayofweek", "month"],
    rolling_windows={7: ["mean", "std"], 30: ["mean"]},
    known_covariates=["holiday"],
    observed_covariates=["promotion"],
)

factory = FeatureFactory(config)
feature_matrix = factory.create_features(dataset)

print(f"Features created: {len(feature_matrix.feature_cols)}")
print(f"Feature hash: {feature_matrix.config_hash}")
print(f"Signature: {feature_matrix.signature}")
```

---

*Document Version: 0.2.0*
*Last Updated: 2026-01-31*
*Author: Development Team*
