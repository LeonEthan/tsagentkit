"""Models module for tsagentkit.

Minimal model fitting and prediction using the registry/cache/protocol pattern.
"""

from tsagentkit.models.cache import ModelCache
from tsagentkit.models.ensemble import ensemble, ensemble_with_quantiles
from tsagentkit.models.length_utils import (
    LengthAdjustment,
    adjust_context_length,
    check_data_compatibility,
    get_effective_limits,
    validate_prediction_length,
)
from tsagentkit.models.output_utils import (
    extract_batch_forecasts,
    extract_batch_quantiles,
    extract_point_forecast,
    extract_predictions_array,
    resolve_quantile_index,
    select_median_index,
    tensor_to_numpy,
)
from tsagentkit.models.protocol import fit, predict
from tsagentkit.models.registry import (
    REGISTRY,
    ModelSpec,
    get_spec,
    list_models,
)

__all__ = [
    # Registry
    "REGISTRY",
    "ModelSpec",
    "list_models",
    "get_spec",
    # Cache
    "ModelCache",
    # Protocol
    "fit",
    "predict",
    # Ensemble
    "ensemble",
    "ensemble_with_quantiles",
    # Length utilities
    "LengthAdjustment",
    "adjust_context_length",
    "validate_prediction_length",
    "get_effective_limits",
    "check_data_compatibility",
    # Output utilities
    "tensor_to_numpy",
    "extract_predictions_array",
    "resolve_quantile_index",
    "select_median_index",
    "extract_point_forecast",
    "extract_batch_forecasts",
    "extract_batch_quantiles",
]
