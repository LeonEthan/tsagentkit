"""Salesforce Moirai 2.0 TSFM adapter.

Adapter for Salesforce's Moirai 2.0 universal time series forecasting model.
Moirai 2.0 is a transformer-based model with improved architecture.

Reference: https://github.com/SalesforceAIResearch/uni2ts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tsagentkit.time import normalize_pandas_freq
from tsagentkit.utils import quantile_col_name

from .base import TSFMAdapter, _timed_model_load

if TYPE_CHECKING:
    from tsagentkit.contracts import ForecastResult
    from tsagentkit.series import TSDataset


class MoiraiAdapter(TSFMAdapter):
    """Adapter for Salesforce Moirai 2.0 foundation model.

    Moirai 2.0 is a universal time series forecasting transformer with
    improved architecture over Moirai 1.x.

    Available model sizes:
        - small: 384d model, 6 layers (recommended for fast inference)

    Example:
        >>> config = AdapterConfig(model_name="moirai", model_size="small")
        >>> adapter = MoiraiAdapter(config)
        >>> adapter.load_model()
        >>> result = adapter.predict(dataset, horizon=30)

    Reference:
        https://github.com/SalesforceAIResearch/uni2ts
    """

    # HuggingFace model ID for Moirai 2.0 (only small available currently)
    MODEL_ID = "Salesforce/moirai-2.0-R-small"

    # Default context length for Moirai 2.0
    DEFAULT_CONTEXT_LENGTH = 512

    @_timed_model_load
    def load_model(self) -> None:
        """Load Moirai 2.0 model from HuggingFace.

        Downloads and caches the model if not already present.

        Raises:
            ImportError: If uni2ts is not installed
            RuntimeError: If model loading fails
        """
        try:
            from uni2ts.model.moirai2 import Moirai2Module
        except ImportError as e:
            raise ImportError(
                "uni2ts>=2.0.0 is required for MoiraiAdapter. "
                "Install with: pip install 'uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git'"
            ) from e

        self._module = Moirai2Module.from_pretrained(self.MODEL_ID)
        self._model = self._module

    def _prepare_model(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """Moirai requires no additional preparation (zero-shot)."""
        return {}

    def _get_model_name(self) -> str:
        """Return model name for Moirai."""
        return "moirai-2.0"

    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using Moirai 2.0.

        Args:
            dataset: Historical data for context
            horizon: Number of steps to forecast
            quantiles: Quantile levels for probabilistic forecasts

        Returns:
            ForecastResult with predictions and provenance
        """
        self._require_loaded("predict")

        try:
            from gluonts.dataset.common import ListDataset
            from uni2ts.model.moirai2 import Moirai2Forecast
        except ImportError as e:
            raise ImportError(
                "gluonts and uni2ts>=2.0.0 are required for MoiraiAdapter."
            ) from e

        from tsagentkit.contracts import ForecastResult

        freq = normalize_pandas_freq(dataset.freq)
        context_length = self._get_context_length(dataset, horizon)

        model = Moirai2Forecast(
            module=self._module,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=self.config.prediction_batch_size or 32)

        entries = []
        meta = []
        for uid in dataset.series_ids:
            series_df = dataset.get_series(uid).sort_values("ds")
            values = series_df["y"].values.astype(np.float32)
            if np.any(np.isnan(values)):
                values = self._handle_missing_values(values)
            entries.append(
                {
                    "item_id": uid,
                    "start": series_df["ds"].iloc[0],
                    "target": values,
                }
            )
            meta.append({"uid": uid, "last_date": series_df["ds"].max()})

        gluonts_ds = ListDataset(entries, freq=freq)
        forecast_it = predictor.predict(gluonts_ds)

        offset = pd.tseries.frequencies.to_offset(freq)
        result_rows = []
        for meta_item, forecast in zip(meta, forecast_it, strict=False):
            uid = meta_item["uid"]
            last_date = meta_item["last_date"]
            future_dates = pd.date_range(
                start=last_date + offset,
                periods=horizon,
                freq=freq,
            )

            point_forecast = (
                forecast.quantile(0.5) if quantiles and 0.5 in quantiles else forecast.mean
            )
            point_forecast = np.asarray(point_forecast).flatten()

            quantile_arrays: dict[float, np.ndarray] = {}
            if quantiles:
                for q in quantiles:
                    try:
                        quantile_arrays[q] = np.asarray(forecast.quantile(q)).flatten()
                    except Exception:
                        quantile_arrays[q] = point_forecast

            for h in range(horizon):
                row = {
                    "unique_id": uid,
                    "ds": future_dates[h],
                    "yhat": float(point_forecast[h]),
                }
                for q in quantiles or []:
                    row[quantile_col_name(q)] = float(quantile_arrays[q][h])
                result_rows.append(row)

        result_df = pd.DataFrame(result_rows)
        result_df["model"] = "moirai-2.0"
        provenance = self._create_provenance(dataset, horizon, quantiles)

        return ForecastResult(
            df=result_df,
            provenance=provenance,
            model_name="moirai-2.0",
            horizon=horizon,
        )

    def _get_context_length(self, dataset: TSDataset, horizon: int) -> int:
        """Get appropriate context length for prediction.

        Args:
            dataset: Input dataset
            horizon: Forecast horizon

        Returns:
            Context length (capped at model max)
        """
        max_series_len = (
            int(dataset.df.groupby("unique_id").size().max()) if not dataset.df.empty else 0
        )

        context_length = self.config.max_context_length or max_series_len
        context_length = max(context_length, horizon)
        context_length = min(context_length, self.DEFAULT_CONTEXT_LENGTH)
        return max(1, int(context_length))

    def get_model_signature(self) -> str:
        """Return model signature for provenance."""
        return f"moirai-2.0-{self._device}"

    @classmethod
    def _check_dependencies_impl(cls) -> None:
        """Check if Moirai dependencies are installed."""
        try:
            from uni2ts.model.moirai2 import Moirai2Module  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "uni2ts>=2.0.0 is required. "
                "Install with: pip install 'uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git'"
            ) from e

    @classmethod
    def _get_capability_spec(cls, adapter_name: str) -> dict[str, Any]:
        return {
            "adapter_name": adapter_name,
            "provider": "salesforce",
            "is_zero_shot": True,
            "supports_quantiles": True,
            "supports_past_covariates": False,
            "supports_future_covariates": False,
            "supports_static_covariates": False,
            "max_context_length": cls.DEFAULT_CONTEXT_LENGTH,
            "max_horizon": None,
            "dependencies": ["torch", "uni2ts>=2.0.0", "gluonts"],
            "notes": "Moirai 2.0 adapter currently consumes target-only panel context.",
        }
