# tsagentkit v1.0 Development Plan

> **Document Goal**: Technical implementation plan for v1.0 "Ecosystem"
> **Target Audience**: Core developers, AI agents contributing code
> **Status**: Draft - Ready for implementation

---

## 1. Overview

### 1.1 Scope

This plan defines the implementation details for v1.0: **Ecosystem**. The goal is to extend tsagentkit into a production-ready ecosystem that integrates with external Time-Series Foundation Models (TSFMs) and supports hierarchical forecasting with reconciliation capabilities.

### 1.2 Success Criteria

- [ ] External adapter system supports at least 3 TSFMs (Chronos, Moirai, TimesFM)
- [ ] Unified adapter interface for TSFMs with automatic model downloading/caching
- [ ] Hierarchical reconciliation engine supports bottom-up, top-down, and middle-out methods
- [ ] Optimal reconciliation (MinT) implemented using `hierarchicalforecast`
- [ ] Hierarchy metadata properly integrated with `TSDataset` and provenance
- [ ] Cross-validation with reconciliation in backtesting workflow
- [ ] All v0.2 tests continue to pass (backward compatibility)
- [ ] 500+ tests passing with 95%+ coverage

---

## 2. Module Implementation Plan

### 2.1 Project Structure

```
tagentkit/
├── src/tsagentkit/
│   ├── ... (v0.1 and v0.2 modules)
│   ├── models/
│   │   ├── ... (existing v0.1 files)
│   │   └── adapters/                # NEW: TSFM adapters
│   │       ├── __init__.py
│   │       ├── base.py              # Abstract TSFM adapter interface
│   │       ├── chronos.py           # Amazon Chronos adapter
│   │       ├── moirai.py            # Salesforce Moirai adapter
│   │       ├── timesfm.py           # Google TimesFM adapter
│   │       └── registry.py          # Adapter registry and discovery
│   └── hierarchy/                   # NEW: Hierarchical reconciliation
│       ├── __init__.py
│       ├── structure.py             # Hierarchy structure definition
│       ├── aggregation.py           # Aggregation matrix operations
│       ├── reconciliation.py        # Reconciliation methods (bottom-up, MinT)
│       ├── evaluator.py             # Hierarchy-aware evaluation
│       └── utils.py                 # Hierarchy validation utilities
├── tests/
│   ├── ... (existing tests)
│   ├── models/adapters/
│   │   ├── test_chronos.py
│   │   ├── test_moirai.py
│   │   ├── test_timesfm.py
│   │   └── test_registry.py
│   └── hierarchy/
│       ├── test_structure.py
│       ├── test_aggregation.py
│       ├── test_reconciliation.py
│       └── test_evaluator.py
└── docs/
    ├── PRD.md
    ├── DEV_PLAN_v0.1.md
    ├── DEV_PLAN_v0.2.md
    └── DEV_PLAN_v1.0.md (this file)
```

---

## 3. Phase-by-Phase Implementation

### Phase 1: External TSFM Adapters (Week 1-2)

**Goal**: Build a pluggable adapter system for external Time-Series Foundation Models

#### 3.1.1 New Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
# Existing groups...
tsfm = [
    "chronos-forecasting>=0.1.0",    # Amazon Chronos
    "moirai-pytorch>=0.1.0",         # Salesforce Moirai
    "timesfm>=0.1.0",                # Google TimesFM (when available)
    "torch>=2.0.0",                  # Required by most TSFMs
    "transformers>=4.30.0",          # HuggingFace integration
]
hierarchy = [
    "hierarchicalforecast>=0.4.0",   # Nixtla reconciliation
]
```

#### 3.1.2 Adapter Base Module (`models/adapters/`)

**File: `models/adapters/base.py`**
- `TSFMAdapter` abstract base class
- Unified interface for all TSFMs
- Automatic device management (CPU/GPU)
- Model caching and versioning

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
import torch


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for TSFM adapter."""

    model_name: str
    model_size: Literal["small", "base", "large"] = "base"
    device: str | None = None  # Auto-detect if None
    cache_dir: str | None = None
    batch_size: int = 32
    prediction_batch_size: int = 100


class TSFMAdapter(ABC):
    """Abstract base class for Time-Series Foundation Model adapters.

    Provides unified interface for different TSFMs while handling
    model-specific quirks internally.
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._model: Any | None = None
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        """Resolve compute device (cuda/mps/cpu)."""
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @abstractmethod
    def load_model(self) -> None:
        """Load the foundation model (with caching)."""
        pass

    @abstractmethod
    def fit(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ModelArtifact:
        """Zero-shot or fine-tuned prediction on dataset.

        Note: Most TSFMs are zero-shot; this may just validate
        compatibility and prepare the model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using the TSFM.

        Args:
            dataset: Historical data
            horizon: Number of steps to forecast
            quantiles: Optional quantile levels for probabilistic forecasts

        Returns:
            ForecastResult with predictions and provenance
        """
        pass

    @abstractmethod
    def get_model_signature(self) -> str:
        """Return unique signature for this model configuration."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._model is not None
```

#### 3.1.3 Chronos Adapter (`models/adapters/chronos.py`)

**Integration**: `chronos-forecasting` from Amazon

```python
class ChronosAdapter(TSFMAdapter):
    """Adapter for Amazon Chronos time series models.

    Chronos is a family of pretrained time series forecasting models
    based on T5 architecture. It supports zero-shot forecasting.

    Reference: https://github.com/amazon-science/chronos-forecasting
    """

    MODEL_SIZES = {
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
        "tiny": "amazon/chronos-t5-tiny",
    }

    def load_model(self) -> None:
        """Load Chronos model from HuggingFace."""
        from chronos import ChronosPipeline

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["base"]
        )

        self._model = ChronosPipeline.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            device_map=self._device,
        )

    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using Chronos.

        Chronos natively supports quantile prediction via sampling.
        """
        if not self.is_loaded:
            self.load_model()

        # Convert TSDataset to Chronos format
        context = self._to_chronos_format(dataset)

        # Generate predictions
        # Chronos uses sampling for quantiles
        num_samples = 100 if quantiles else 1

        forecasts = []
        for batch in self._batch_iterator(context, self.config.batch_size):
            batch_forecast = self._model.predict(
                context=batch,
                prediction_length=horizon,
                num_samples=num_samples,
            )
            forecasts.append(batch_forecast)

        # Convert to ForecastResult format
        return self._to_forecast_result(
            forecasts, dataset, horizon, quantiles
        )

    def _to_chronos_format(self, dataset: TSDataset) -> list:
        """Convert TSDataset to Chronos tensor format."""
        import torch

        series_list = []
        for uid in dataset.unique_ids:
            series = dataset.get_series(uid)["y"].values
            series_list.append(torch.tensor(series, dtype=torch.float32))

        return series_list

    def get_model_signature(self) -> str:
        """Return model signature for provenance."""
        return f"chronos-{self.config.model_size}-{self._device}"
```

#### 3.1.4 Moirai Adapter (`models/adapters/moirai.py`)

**Integration**: `moirai-pytorch` from Salesforce

```python
class MoiraiAdapter(TSFMAdapter):
    """Adapter for Salesforce Moirai foundation model.

    Moirai is a universal time series forecasting transformer
    trained on large-scale time series data.

    Reference: https://github.com/SalesforceAIResearch/uni2ts
    """

    MODEL_SIZES = {
        "small": "Salesforce/moirai-1.0-R-small",
        "base": "Salesforce/moirai-1.0-R-base",
        "large": "Salesforce/moirai-1.1-R-large",
    }

    def load_model(self) -> None:
        """Load Moirai model from HuggingFace."""
        from uni2ts.model.moirai import MoiraiForecast

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["base"]
        )

        self._model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=model_id,
            map_location=self._device,
        )

    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using Moirai.

        Moirai uses a patch-based architecture for variable-length series.
        """
        if not self.is_loaded:
            self.load_model()

        # Moirai-specific prediction logic
        forecasts = []
        for uid in dataset.unique_ids:
            series = dataset.get_series(uid)
            forecast = self._predict_single_series(series, horizon, quantiles)
            forecasts.append(forecast)

        return self._aggregate_forecasts(forecasts, dataset, horizon, quantiles)

    def _predict_single_series(
        self,
        series: pd.DataFrame,
        horizon: int,
        quantiles: list[float] | None,
    ) -> torch.Tensor:
        """Predict for a single time series."""
        # Implementation details for Moirai prediction
        pass

    def get_model_signature(self) -> str:
        """Return model signature for provenance."""
        return f"moirai-{self.config.model_size}-{self._device}"
```

#### 3.1.5 TimesFM Adapter (`models/adapters/timesfm.py`)

**Integration**: Google TimesFM (when available via pip)

```python
class TimesFMAdapter(TSFMAdapter):
    """Adapter for Google TimesFM foundation model.

    TimesFM is a pretrained time-series foundation model from Google
    designed for zero-shot forecasting.

    Reference: https://github.com/google-research/timesfm
    """

    MODEL_SIZES = {
        "base": "google/timesfm-1.0-200m",
        "large": "google/timesfm-1.0-500m",
    }

    def load_model(self) -> None:
        """Load TimesFM model."""
        import timesfm

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["base"]
        )

        self._model = timesfm.TimesFm(
            context_len=512,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=self._device,
        )
        self._model.load_from_checkpoint(repo_id=model_id)

    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using TimesFM."""
        if not self.is_loaded:
            self.load_model()

        # TimesFM prediction logic
        forecasts = self._model.forecast(
            inputs=self._to_timesfm_format(dataset),
            freq=self._map_frequency(dataset.freq),
        )

        return self._to_forecast_result(forecasts, dataset, horizon, quantiles)

    def _map_frequency(self, freq: str) -> str:
        """Map pandas frequency to TimesFM frequency format."""
        freq_map = {
            "D": "D",
            "H": "H",
            "W": "W",
            "M": "M",
            "Q": "Q",
            "Y": "Y",
        }
        return freq_map.get(freq[0].upper(), "D")

    def get_model_signature(self) -> str:
        """Return model signature for provenance."""
        return f"timesfm-{self.config.model_size}-{self._device}"
```

#### 3.1.6 Adapter Registry (`models/adapters/registry.py`)

```python
class AdapterRegistry:
    """Registry for TSFM adapters with auto-discovery.

    Provides centralized access to available adapters with
    fallback handling and version checking.
    """

    _adapters: dict[str, type[TSFMAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[TSFMAdapter]) -> None:
        """Register an adapter class."""
        cls._adapters[name] = adapter_class

    @classmethod
    def get(cls, name: str) -> type[TSFMAdapter]:
        """Get adapter class by name."""
        if name not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
        return cls._adapters[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered adapter names."""
        return list(cls._adapters.keys())

    @classmethod
    def create(
        cls,
        name: str,
        config: AdapterConfig | None = None,
    ) -> TSFMAdapter:
        """Factory method to create adapter instance."""
        adapter_class = cls.get(name)
        return adapter_class(config or AdapterConfig(model_name=name))

    @classmethod
    def check_availability(cls, name: str) -> tuple[bool, str | None]:
        """Check if adapter dependencies are installed.

        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            adapter_class = cls.get(name)
            # Try to import dependencies
            adapter_class._check_dependencies()
            return True, None
        except ImportError as e:
            return False, str(e)
        except ValueError as e:
            return False, str(e)


# Register built-in adapters
AdapterRegistry.register("chronos", ChronosAdapter)
AdapterRegistry.register("moirai", MoiraiAdapter)
AdapterRegistry.register("timesfm", TimesFMAdapter)
```

#### Deliverables
- [ ] `TSFMAdapter` abstract base class with unified interface
- [ ] `ChronosAdapter` for Amazon Chronos models
- [ ] `MoiraiAdapter` for Salesforce Moirai models
- [ ] `TimesFMAdapter` for Google TimesFM models
- [ ] `AdapterRegistry` for discovery and factory creation
- [ ] Automatic device detection (CUDA/MPS/CPU)
- [ ] Model caching and version management
- [ ] Unit tests for each adapter with mocked models (30 tests)
- [ ] Integration tests for adapter registry (10 tests)

---

### Phase 2: Hierarchical Reconciliation (Week 2-3)

**Goal**: Implement hierarchical forecasting with multiple reconciliation strategies

#### 3.2.1 Hierarchy Module (`hierarchy/`)

**File: `hierarchy/structure.py`**
- `HierarchyStructure` dataclass for defining aggregation relationships
- Support for multiple hierarchy types (tree, DAG for grouped structures)
- Validation of hierarchy consistency

```python
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchyStructure:
    """Defines hierarchical relationships between time series.

    Represents the aggregation structure where bottom-level series
    sum up to higher-level series.

    Example structure for retail:
        Total
        ├── Region_North
        │   ├── Store_A
        │   └── Store_B
        └── Region_South
            ├── Store_C
            └── Store_D
    """

    # Mapping from parent to children
    aggregation_graph: dict[str, list[str]]

    # Bottom-level nodes (leaf nodes)
    bottom_nodes: list[str]

    # All nodes in the hierarchy
    all_nodes: list[str] = field(init=False)

    # Aggregation matrix S (numpy array)
    # Shape: (n_total, n_bottom)
    # where S[i, j] = 1 if bottom node j contributes to node i
    s_matrix: np.ndarray = field(repr=False)

    def __post_init__(self):
        # Validate structure
        object.__setattr__(
            self,
            "all_nodes",
            self._compute_all_nodes()
        )
        self._validate_structure()

    def _compute_all_nodes(self) -> list[str]:
        """Compute list of all nodes."""
        nodes = set(self.bottom_nodes)
        for parent, children in self.aggregation_graph.items():
            nodes.add(parent)
            nodes.update(children)
        return sorted(nodes)

    def _validate_structure(self) -> None:
        """Validate hierarchy structure is consistent."""
        # Check all children exist
        for parent, children in self.aggregation_graph.items():
            for child in children:
                if child not in self.all_nodes:
                    raise ValueError(f"Child '{child}' not found in hierarchy")

        # Check bottom nodes have no children
        for node in self.bottom_nodes:
            if node in self.aggregation_graph:
                raise ValueError(f"Bottom node '{node}' cannot have children")

        # Check S matrix dimensions
        n_total = len(self.all_nodes)
        n_bottom = len(self.bottom_nodes)
        if self.s_matrix.shape != (n_total, n_bottom):
            raise ValueError(
                f"S matrix shape {self.s_matrix.shape} doesn't match "
                f"expected ({n_total}, {n_bottom})"
            )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        hierarchy_columns: list[str],
        value_column: str = "y",
    ) -> "HierarchyStructure":
        """Build hierarchy structure from DataFrame with hierarchical columns.

        Args:
            df: DataFrame with hierarchical identifiers
            hierarchy_columns: Columns defining hierarchy (top to bottom)
            value_column: Column containing values

        Example:
            df with columns [country, state, city, y]
            hierarchy_columns = ["country", "state", "city"]
        """
        # Build aggregation graph from unique combinations
        pass

    @classmethod
    def from_summation_matrix(
        cls,
        s_matrix: np.ndarray,
        node_names: list[str],
        bottom_node_names: list[str],
    ) -> "HierarchyStructure":
        """Build hierarchy from explicit summation matrix."""
        pass

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes of a given node."""
        parents = []
        for parent, children in self.aggregation_graph.items():
            if node in children:
                parents.append(parent)
        return parents

    def get_children(self, node: str) -> list[str]:
        """Get child nodes of a given node."""
        return self.aggregation_graph.get(node, [])

    def get_level(self, node: str) -> int:
        """Get hierarchy level (0 = root, increasing downward)."""
        if node not in self.all_nodes:
            raise ValueError(f"Node '{node}' not in hierarchy")

        level = 0
        current = node
        parents = self.get_parents(current)

        while parents:
            level += 1
            current = parents[0]  # Assume tree structure for level
            parents = self.get_parents(current)

        return level
```

**File: `hierarchy/aggregation.py`**
- Aggregation matrix operations
- Helper functions for common hierarchy patterns

```python
def create_bottom_up_matrix(structure: HierarchyStructure) -> np.ndarray:
    """Create projection matrix for bottom-up reconciliation.

    Bottom-up simply takes bottom-level forecasts and aggregates them
    using the S matrix.

    Returns:
        Projection matrix P such that ŷ_reconciled = S @ P @ ŷ_base
        For bottom-up, P selects bottom-level series.
    """
    n_bottom = len(structure.bottom_nodes)
    n_total = len(structure.all_nodes)

    # P extracts bottom-level forecasts
    p_matrix = np.zeros((n_bottom, n_total))
    for i, node in enumerate(structure.bottom_nodes):
        idx = structure.all_nodes.index(node)
        p_matrix[i, idx] = 1

    return p_matrix


def create_top_down_matrix(
    structure: HierarchyStructure,
    proportions: dict[str, float] | None = None,
) -> np.ndarray:
    """Create projection matrix for top-down reconciliation.

    Distributes top-level forecasts down the hierarchy using
    historical proportions.
    """
    pass


def create_middle_out_matrix(
    structure: HierarchyStructure,
    middle_level: int,
) -> np.ndarray:
    """Create projection matrix for middle-out reconciliation.

    Uses bottom-up from middle level downward, top-down from
    middle level upward.
    """
    pass
```

**File: `hierarchy/reconciliation.py`**
- Reconciliation methods: bottom-up, top-down, middle-out, OLS, MinT

```python
from enum import Enum, auto
from typing import Callable
import numpy as np
from scipy.linalg import inv, solve


class ReconciliationMethod(Enum):
    """Available reconciliation methods."""

    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    MIDDLE_OUT = "middle_out"
    OLS = "ols"  # Ordinary Least Squares (simple average)
    MIN_TRACE = "mint"  # Minimum Trace optimal reconciliation


class Reconciler:
    """Hierarchical forecast reconciliation engine.

    Implements various reconciliation strategies to ensure
    coherent forecasts across the hierarchy.
    """

    def __init__(
        self,
        method: ReconciliationMethod,
        structure: HierarchyStructure,
    ):
        self.method = method
        self.structure = structure
        self._projection_matrix: np.ndarray | None = None

    def reconcile(
        self,
        base_forecasts: np.ndarray,
        fitted_values: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reconcile base forecasts to be hierarchy-consistent.

        Args:
            base_forecasts: Array of shape (n_nodes, horizon)
            fitted_values: Fitted values for MinT (n_nodes, n_obs)
            residuals: Residuals for MinT variance estimation

        Returns:
            Reconciled forecasts of shape (n_nodes, horizon)
        """
        if self._projection_matrix is None:
            self._projection_matrix = self._compute_projection_matrix(
                fitted_values, residuals
            )

        # Reconcile: ŷ_reconciled = S @ P @ ŷ_base
        return self.structure.s_matrix @ self._projection_matrix @ base_forecasts

    def _compute_projection_matrix(
        self,
        fitted_values: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute projection matrix for the reconciliation method."""
        method_map: dict[ReconciliationMethod, Callable] = {
            ReconciliationMethod.BOTTOM_UP: self._bottom_up_matrix,
            ReconciliationMethod.TOP_DOWN: self._top_down_matrix,
            ReconciliationMethod.MIDDLE_OUT: self._middle_out_matrix,
            ReconciliationMethod.OLS: self._ols_matrix,
            ReconciliationMethod.MIN_TRACE: self._mint_matrix,
        }

        func = method_map[self.method]
        if self.method == ReconciliationMethod.MIN_TRACE:
            return func(fitted_values, residuals)
        return func()

    def _bottom_up_matrix(self) -> np.ndarray:
        """Bottom-up projection matrix."""
        return create_bottom_up_matrix(self.structure)

    def _top_down_matrix(self) -> np.ndarray:
        """Top-down projection matrix."""
        return create_top_down_matrix(self.structure)

    def _middle_out_matrix(self) -> np.ndarray:
        """Middle-out projection matrix."""
        # Default to middle level
        return create_middle_out_matrix(self.structure, middle_level=1)

    def _ols_matrix(self) -> np.ndarray:
        """OLS (structural) projection matrix.

        P_ols = (S' S)^(-1) S'
        """
        s = self.structure.s_matrix
        return inv(s.T @ s) @ s.T

    def _mint_matrix(
        self,
        fitted_values: np.ndarray | None,
        residuals: np.ndarray | None,
    ) -> np.ndarray:
        """MinT (minimum trace) optimal reconciliation.

        Uses estimated variance-covariance matrix of forecast errors
        to produce optimal reconciled forecasts.

        P_mint = (S' W^(-1) S)^(-1) S' W^(-1)

        where W is the variance-covariance matrix of base forecast errors.
        """
        if residuals is None:
            # Fall back to OLS if no residuals provided
            return self._ols_matrix()

        # Estimate W from residuals
        w = self._estimate_w(residuals)

        s = self.structure.s_matrix
        # Solve for P_mint
        p_mint = solve(s.T @ inv(w) @ s, s.T @ inv(w))

        return p_mint

    def _estimate_w(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate variance-covariance matrix from residuals.

        Args:
            residuals: Residuals of shape (n_nodes, n_obs)

        Returns:
            Estimated W matrix of shape (n_nodes, n_nodes)
        """
        # Sample covariance matrix
        return np.cov(residuals)


def reconcile_forecasts(
    base_forecasts: pd.DataFrame,
    structure: HierarchyStructure,
    method: ReconciliationMethod,
    fitted_values: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """High-level function for hierarchical reconciliation.

    Args:
        base_forecasts: DataFrame with columns [unique_id, ds, yhat, ...]
        structure: Hierarchy structure
        method: Reconciliation method
        fitted_values: Optional fitted values for MinT
        residuals: Optional residuals for MinT

    Returns:
        Reconciled forecasts DataFrame
    """
    reconciler = Reconciler(method, structure)

    # Convert DataFrame to matrix format
    # ... transformation logic ...

    reconciled_matrix = reconciler.reconcile(
        base_forecasts_matrix,
        fitted_values_matrix if fitted_values is not None else None,
        residuals_matrix if residuals is not None else None,
    )

    # Convert back to DataFrame
    # ... transformation logic ...

    return reconciled_df
```

**File: `hierarchy/evaluator.py`**
- Hierarchy-aware evaluation metrics
- Coherence violation detection

```python
@dataclass(frozen=True)
class HierarchyEvaluationReport:
    """Evaluation report for hierarchical forecasts."""

    # Per-level metrics
    level_metrics: dict[int, dict[str, float]]

    # Coherence violations
    coherence_violations: list[CoherenceViolation]

    # Overall coherence score (0-1, higher is better)
    coherence_score: float

    # Reconciliation improvement vs base forecasts
    reconciliation_improvement: dict[str, float]


@dataclass(frozen=True)
class CoherenceViolation:
    """Single coherence violation record."""

    parent_node: str
    expected_value: float
    actual_value: float
    difference: float
    timestamp: str


class HierarchyEvaluator:
    """Evaluate hierarchical forecast quality and coherence."""

    def __init__(self, structure: HierarchyStructure):
        self.structure = structure

    def evaluate(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> HierarchyEvaluationReport:
        """Evaluate hierarchical forecasts.

        Computes:
        - Standard forecast metrics per level
        - Coherence violations (where children don't sum to parent)
        - Overall coherence score
        """
        # Compute per-level metrics
        level_metrics = self._compute_level_metrics(forecasts, actuals)

        # Detect coherence violations
        violations = self._detect_violations(forecasts)

        # Compute coherence score
        coherence_score = self._compute_coherence_score(forecasts)

        return HierarchyEvaluationReport(
            level_metrics=level_metrics,
            coherence_violations=violations,
            coherence_score=coherence_score,
            reconciliation_improvement={},
        )

    def _detect_violations(
        self,
        forecasts: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> list[CoherenceViolation]:
        """Detect where forecasts violate hierarchical coherence."""
        violations = []

        for parent, children in self.structure.aggregation_graph.items():
            parent_forecast = forecasts[forecasts["unique_id"] == parent]
            children_sum = (
                forecasts[forecasts["unique_id"].isin(children)]
                .groupby("ds")["yhat"]
                .sum()
            )

            for ds in parent_forecast["ds"]:
                parent_val = parent_forecast[parent_forecast["ds"] == ds]["yhat"].iloc[0]
                children_val = children_sum.get(ds, 0)

                diff = abs(parent_val - children_val)
                if diff > tolerance:
                    violations.append(CoherenceViolation(
                        parent_node=parent,
                        expected_value=parent_val,
                        actual_value=children_val,
                        difference=diff,
                        timestamp=str(ds),
                    ))

        return violations

    def _compute_coherence_score(
        self,
        forecasts: pd.DataFrame,
    ) -> float:
        """Compute overall coherence score.

        Score is 1.0 if perfectly coherent, decreases with violations.
        """
        violations = self._detect_violations(forecasts)
        if not violations:
            return 1.0

        # Score based on magnitude of violations
        total_diff = sum(v.difference for v in violations)
        total_value = forecasts["yhat"].abs().sum()

        return max(0.0, 1.0 - (total_diff / total_value))
```

#### Deliverables
- [ ] `HierarchyStructure` with validation and S matrix support
- [ ] `Reconciler` class with 5 reconciliation methods (bottom-up, top-down, middle-out, OLS, MinT)
- [ ] `HierarchyEvaluator` for coherence checking
- [ ] Integration with `hierarchicalforecast` for advanced methods
- [ ] Support for tree and grouped (DAG) hierarchies
- [ ] Unit tests for all reconciliation methods (40 tests)
- [ ] Integration tests for end-to-end hierarchy workflows (15 tests)

---

### Phase 3: Integration with Core Pipeline (Week 3)

**Goal**: Integrate adapters and hierarchy into the main forecasting pipeline

#### 3.3.1 Enhanced TSDataset for Hierarchy

**File: `series/dataset.py` - Enhancements**

```python
@dataclass(frozen=True)
class TSDataset:
    """Enhanced TSDataset with hierarchy support."""

    # ... existing fields ...

    # Optional hierarchy structure
    hierarchy: HierarchyStructure | None = None

    def with_hierarchy(self, hierarchy: HierarchyStructure) -> "TSDataset":
        """Return new TSDataset with hierarchy attached."""
        return replace(self, hierarchy=hierarchy)

    def is_hierarchical(self) -> bool:
        """Check if dataset has hierarchy."""
        return self.hierarchy is not None

    def get_level_series(self, level: int) -> list[str]:
        """Get all series IDs at a specific hierarchy level."""
        if not self.hierarchy:
            return self.unique_ids

        return [
            node for node in self.hierarchy.all_nodes
            if self.hierarchy.get_level(node) == level
        ]

    def aggregate_to_level(self, target_level: int) -> "TSDataset":
        """Aggregate bottom-level data to target hierarchy level."""
        # Implementation for aggregation
        pass
```

#### 3.3.2 Enhanced Router with TSFM Support

**File: `router/router.py` - Enhancements**

```python
def make_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    qa: QAReport | None = None,
    use_tsfm: bool = True,
    tsfm_preference: list[str] | None = None,
) -> Plan:
    """Enhanced plan generation with TSFM support.

    Args:
        dataset: Time series dataset
        task_spec: Task specification
        qa: Optional QA report
        use_tsfm: Whether to consider TSFMs
        tsfm_preference: Ordered list of preferred TSFMs
    """
    # Check TSFM availability
    available_tsfms = []
    if use_tsfm:
        for tsfm_name in (tsfm_preference or ["chronos", "moirai", "timesfm"]):
            is_avail, _ = AdapterRegistry.check_availability(tsfm_name)
            if is_avail:
                available_tsfms.append(tsfm_name)

    # Hierarchy-aware planning
    if dataset.is_hierarchical():
        return _make_hierarchical_plan(
            dataset, task_spec, qa, available_tsfms
        )

    # Standard planning with TSFM at top of ladder
    return _make_standard_plan(dataset, task_spec, qa, available_tsfms)


def _make_hierarchical_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    qa: QAReport | None,
    available_tsfms: list[str],
) -> Plan:
    """Create plan for hierarchical forecasting."""
    # Determine reconciliation strategy
    n_levels = max(
        dataset.hierarchy.get_level(node)
        for node in dataset.hierarchy.all_nodes
    ) + 1

    # For deep hierarchies, use MinT
    # For shallow hierarchies, bottom-up may suffice
    if n_levels > 2:
        reconciliation_method = ReconciliationMethod.MIN_TRACE
    else:
        reconciliation_method = ReconciliationMethod.BOTTOM_UP

    return Plan(
        primary_model="chronos" if available_tsfms else "seasonal_naive",
        fallback_chain=available_tsfms[1:] + ["theta", "seasonal_naive"],
        config={
            "hierarchical": True,
            "reconciliation_method": reconciliation_method.value,
            "hierarchy_levels": n_levels,
        },
        signature=_compute_plan_signature(dataset, task_spec),
    )
```

#### 3.3.3 Enhanced Backtest with Reconciliation

**File: `backtest/engine.py` - Enhancements**

```python
def rolling_backtest(
    dataset: TSDataset,
    spec: TaskSpec,
    plan: Plan,
    n_windows: int = 5,
    window_strategy: Literal["expanding", "sliding"] = "expanding",
    reconcile: bool = True,
) -> BacktestReport:
    """Enhanced backtest with optional reconciliation.

    Args:
        dataset: Time series dataset
        spec: Task specification
        plan: Execution plan
        n_windows: Number of validation windows
        window_strategy: Window expansion strategy
        reconcile: Whether to apply reconciliation for hierarchical data
    """
    results = []

    for window in generate_windows(dataset, n_windows, window_strategy):
        # Fit and predict
        model = fit(window.train, plan)
        forecast = predict(window.train, model, spec.horizon)

        # Apply reconciliation if hierarchical
        if reconcile and dataset.is_hierarchical():
            forecast = _reconcile_forecast(
                forecast,
                dataset.hierarchy,
                plan.config.get("reconciliation_method", "bottom_up"),
            )

        results.append(evaluate(window.test, forecast))

    return BacktestReport(results=results)
```

#### 3.3.4 Enhanced Serving with TSFM Caching

**File: `serving/inference.py` - Enhancements**

```python
class TSFMCache:
    """Cache for loaded TSFM models to avoid repeated loading."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self._cache: dict[str, TSFMAdapter] = {}
        self._access_times: dict[str, float] = {}

    def get(self, model_name: str, config: AdapterConfig) -> TSFMAdapter:
        """Get or create adapter, with LRU eviction."""
        cache_key = f"{model_name}-{config.model_size}"

        if cache_key in self._cache:
            self._access_times[cache_key] = time.time()
            return self._cache[cache_key]

        # Create new adapter
        adapter = AdapterRegistry.create(model_name, config)
        adapter.load_model()

        # Evict if necessary
        if len(self._cache) >= self.max_size:
            lru_key = min(self._access_times, key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]

        self._cache[cache_key] = adapter
        self._access_times[cache_key] = time.time()

        return adapter


# Global cache instance
_tsfm_cache = TSFMCache()


def predict(
    dataset: TSDataset,
    artifact: ModelArtifact,
    spec: TaskSpec,
    use_cache: bool = True,
) -> ForecastResult:
    """Enhanced prediction with TSFM caching."""
    if artifact.model_type.startswith("tsfm-"):
        model_name = artifact.model_type.replace("tsfm-", "")
        config = AdapterConfig(
            model_name=model_name,
            model_size=artifact.config.get("model_size", "base"),
        )

        if use_cache:
            adapter = _tsfm_cache.get(model_name, config)
        else:
            adapter = AdapterRegistry.create(model_name, config)
            adapter.load_model()

        return adapter.predict(dataset, spec.horizon, spec.quantiles)

    # Standard prediction path
    return _standard_predict(dataset, artifact, spec)
```

#### Deliverables
- [ ] `TSDataset` hierarchy support
- [ ] TSFM-aware router with adapter registry integration
- [ ] Hierarchical backtesting with reconciliation
- [ ] TSFM model caching for serving
- [ ] Integration tests for complete workflows (20 tests)

---

### Phase 4: Documentation and Recipes (Week 4)

**Goal**: Complete agent-facing documentation for v1.0 features

#### 3.4.1 Skill Documentation Updates

**File: `skill/recipes.md` - Additions**

```markdown
## Recipe 3: TSFM Zero-Shot Forecasting

Demonstrates using Chronos for zero-shot forecasting.

```python
from tsagentkit import run_forecast, TaskSpec
from tsagentkit.models.adapters import AdapterConfig

# Configure TSFM
tsfm_config = AdapterConfig(
    model_name="chronos",
    model_size="base",  # small, base, or large
)

# Run with TSFM as primary model
result = run_forecast(
    data=sales_data,
    task_spec=TaskSpec(horizon=14, freq="D"),
    tsfm_config=tsfm_config,
)
```

## Recipe 4: Hierarchical Sales Forecasting

Demonstrates hierarchical forecasting with reconciliation.

```python
from tsagentkit import run_forecast, TaskSpec
from tsagentkit.hierarchy import HierarchyStructure

# Define hierarchy structure
hierarchy = HierarchyStructure.from_dataframe(
    sales_data,
    hierarchy_columns=["region", "state", "store"],
)

# Run hierarchical forecast
result = run_forecast(
    data=sales_data,
    task_spec=TaskSpec(horizon=7, freq="D"),
    hierarchy=hierarchy,
    reconciliation_method="mint",  # MinT optimal reconciliation
)

# Access coherent forecasts at all levels
print(result.forecasts)  # All levels reconciled
```
```

#### 3.4.2 API Reference Documentation

**File: `skill/API_REFERENCE.md` (new)**

Complete API reference for:
- TSFM Adapters (Chronos, Moirai, TimesFM)
- Hierarchy structures and reconciliation
- Integration with existing modules

#### Deliverables
- [ ] TSFM usage recipe
- [ ] Hierarchical forecasting recipe
- [ ] Complete API reference
- [ ] Migration guide from v0.2 to v1.0

---

## 4. Implementation Summary

### v1.0 New Modules

| Module | Files | Key Classes |
|--------|-------|-------------|
| `models/adapters/` | 5 files | `TSFMAdapter`, `ChronosAdapter`, `MoiraiAdapter`, `TimesFMAdapter`, `AdapterRegistry` |
| `hierarchy/` | 5 files | `HierarchyStructure`, `Reconciler`, `HierarchyEvaluator`, `ReconciliationMethod` |

### New Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `tsfm` | `chronos-forecasting`, `moirai-pytorch`, `timesfm`, `torch` | TSFM adapters |
| `hierarchy` | `hierarchicalforecast` | Reconciliation algorithms |

### Test Summary

| Module | Tests |
|--------|-------|
| TSFM Adapters | 40 tests |
| Adapter Registry | 10 tests |
| Hierarchy Structure | 20 tests |
| Reconciliation | 40 tests |
| Hierarchy Evaluation | 15 tests |
| Integration | 20 tests |
| **Total** | **145+ new tests** |

### API Additions

```python
from tsagentkit import (
    # TSFM Adapters
    AdapterConfig,
    AdapterRegistry,
)

from tsagentkit.models.adapters import (
    TSFMAdapter,
    ChronosAdapter,
    MoiraiAdapter,
    TimesFMAdapter,
)

from tsagentkit.hierarchy import (
    HierarchyStructure,
    Reconciler,
    ReconciliationMethod,
    HierarchyEvaluator,
    reconcile_forecasts,
)
```

---

## 5. Key Technical Decisions

### 5.1 TSFM Adapter Design

| Decision | Rationale |
|----------|-----------|
| Abstract base class | Unified interface across different TSFMs |
| Lazy loading | Models only loaded when needed (memory efficiency) |
| Device auto-detection | Optimal performance without configuration |
| Model caching | Avoid repeated loading in serving scenarios |
| Registry pattern | Easy extension with new TSFMs |

### 5.2 Hierarchy Reconciliation Strategy

| Method | Use Case | Complexity |
|--------|----------|------------|
| Bottom-up | Deep hierarchies, stable bottom series | Low |
| Top-down | Strong top-level signal | Low |
| Middle-out | Mixed signal strength | Medium |
| OLS | Balanced hierarchy | Medium |
| MinT | Optimal reconciliation, requires residuals | High |

### 5.3 Integration Points

| Component | Integration |
|-----------|-------------|
| TSDataset | Optional `hierarchy` field |
| Router | TSFM at top of fallback ladder |
| Backtest | Reconciliation after each window |
| Serving | TSFM cache for model reuse |
| Provenance | Model signatures include TSFM version |

---

## 6. Milestones and Timeline

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | TSFM Adapters - Base | `TSFMAdapter` ABC, `ChronosAdapter`, registry |
| 1-2 | TSFM Adapters - Models | `MoiraiAdapter`, `TimesFMAdapter`, caching |
| 2 | Hierarchy - Structure | `HierarchyStructure`, aggregation matrices |
| 2-3 | Hierarchy - Reconciliation | `Reconciler` with 5 methods, `HierarchyEvaluator` |
| 3 | Integration | TSDataset hierarchy, router, backtest updates |
| 4 | Documentation | Recipes, API reference, migration guide |

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TSFM package availability | High | Graceful fallback, mock for testing |
| GPU memory exhaustion | Medium | Batch processing, automatic CPU fallback |
| Large hierarchy computation | Medium | Sparse matrix operations, caching |
| MinT numerical stability | Medium | Regularization, fallback to OLS |
| API compatibility | Low | All v0.2 APIs remain unchanged |

---

## 8. Post-v1.0 Roadmap

Items deferred to future versions:
- Online learning with TSFMs
- Cross-hierarchical attention mechanisms
- Distributed training for large hierarchies
- Probabilistic reconciliation methods
- Custom TSFM fine-tuning interface

---

## Appendix A: Usage Examples

### A1: TSFM Zero-Shot Forecasting

```python
from tsagentkit import run_forecast, TaskSpec
from tsagentkit.models.adapters import AdapterConfig, AdapterRegistry

# Check available adapters
print(AdapterRegistry.list_available())
# ['chronos', 'moirai', 'timesfm']

# Configure and run
config = AdapterConfig(model_name="chronos", model_size="base")
result = run_forecast(
    data=retail_data,
    task_spec=TaskSpec(horizon=28, freq="D"),
    tsfm_config=config,
)
```

### A2: Hierarchical Forecasting

```python
from tsagentkit import run_forecast
from tsagentkit.hierarchy import HierarchyStructure, ReconciliationMethod

# Build hierarchy
hierarchy = HierarchyStructure.from_dataframe(
    df=sales_data,
    hierarchy_columns=["country", "region", "store"],
)

# Run with MinT reconciliation
result = run_forecast(
    data=sales_data,
    task_spec=task_spec,
    hierarchy=hierarchy,
    reconciliation_method=ReconciliationMethod.MIN_TRACE,
)

# Evaluate coherence
evaluator = HierarchyEvaluator(hierarchy)
report = evaluator.evaluate(result.forecasts, actuals)
print(f"Coherence score: {report.coherence_score:.3f}")
```

### A3: Custom TSFM Adapter

```python
from tsagentkit.models.adapters import TSFMAdapter, AdapterConfig, AdapterRegistry

class MyCustomAdapter(TSFMAdapter):
    def load_model(self):
        # Load custom model
        pass

    def predict(self, dataset, horizon, quantiles=None):
        # Generate predictions
        pass

    def get_model_signature(self):
        return "custom-model-v1"

# Register and use
AdapterRegistry.register("custom", MyCustomAdapter)
```

---

*Document Version: 1.0.0*
*Last Updated: 2026-02-01*
*Author: Development Team*
