"""Amazon Chronos TSFM adapter.

Adapter for Amazon's Chronos time series forecasting models.
Chronos is a family of pretrained models based on T5 architecture
that supports zero-shot forecasting.

Reference: https://github.com/amazon-science/chronos-forecasting
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import TSFMAdapter
from tsagentkit.utils import quantile_col_name

if TYPE_CHECKING:
    from tsagentkit.contracts import ForecastResult, ModelArtifact
    from tsagentkit.series import TSDataset


class ChronosAdapter(TSFMAdapter):
    """Adapter for Amazon Chronos time series models.

    Chronos is a family of pretrained time series forecasting models
    based on T5 architecture. It supports zero-shot forecasting on
    unseen time series data.

    Available model sizes:
        - tiny: Smallest, fastest model for quick prototyping
        - small: Small model with good speed/accuracy tradeoff
        - base: Balanced model (recommended default)
        - large: Most capable model, slower inference

    Example:
        >>> config = AdapterConfig(model_name="chronos", model_size="base")
        >>> adapter = ChronosAdapter(config)
        >>> adapter.load_model()  # Downloads if needed
        >>> result = adapter.predict(dataset, horizon=30)

    Reference:
        https://github.com/amazon-science/chronos-forecasting
    """

    # HuggingFace model IDs for each size
    MODEL_SIZES = {
        "tiny": "amazon/chronos-t5-tiny",
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
    }

    # Maximum context lengths by model size
    MAX_CONTEXT_LENGTHS = {
        "tiny": 512,
        "small": 512,
        "base": 512,
        "large": 512,
    }

    def load_model(self) -> None:
        """Load Chronos model from HuggingFace.

        Downloads and caches the model if not already present.

        Raises:
            ImportError: If chronos-forecasting is not installed
            RuntimeError: If model loading fails
        """
        try:
            from chronos import ChronosPipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting is required for ChronosAdapter. "
                "Install with: pip install chronos-forecasting"
            ) from e

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["base"]
        )

        try:
            self._model = ChronosPipeline.from_pretrained(
                model_id,
                cache_dir=self.config.cache_dir,
                device_map=self._device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos model: {e}") from e

    def fit(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ModelArtifact:
        """Prepare Chronos for prediction.

        Chronos is a zero-shot model and doesn't require training.
        This method validates compatibility and returns a ModelArtifact.

        Args:
            dataset: Dataset to validate
            prediction_length: Forecast horizon
            quantiles: Optional quantile levels

        Returns:
            ModelArtifact with model reference
        """
        from tsagentkit.contracts import ModelArtifact

        if not self.is_loaded:
            self.load_model()

        # Validate dataset
        self._validate_dataset(dataset)

        # Check horizon against model limits
        max_context = self.MAX_CONTEXT_LENGTHS.get(
            self.config.model_size, 512
        )
        if prediction_length > max_context:
            raise ValueError(
                f"Prediction length {prediction_length} exceeds maximum "
                f"context length {max_context} for {self.config.model_size} model"
            )

        return ModelArtifact(
            model=self._model,
            model_name=f"chronos-{self.config.model_size}",
            config={
                "model_size": self.config.model_size,
                "device": self._device,
                "prediction_length": prediction_length,
                "quantiles": quantiles,
            },
        )

    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using Chronos.

        Chronos natively supports quantile prediction via sampling.
        Generates samples and computes quantiles from them.

        Args:
            dataset: Historical data for context
            horizon: Number of steps to forecast
            quantiles: Quantile levels (e.g., [0.1, 0.5, 0.9])

        Returns:
            ForecastResult with predictions and provenance
        """
        if not self.is_loaded:
            self.load_model()

        # Convert to Chronos format
        context_list = self._to_chronos_format(dataset)

        # Generate predictions
        num_samples = self.config.num_samples if quantiles else 1
        all_forecasts = []

        for batch in self._batch_iterator(
            context_list, self.config.prediction_batch_size
        ):
            # Chronos predict returns samples
            forecast_samples = self._model.predict(
                context=batch,
                prediction_length=horizon,
                num_samples=num_samples,
            )
            all_forecasts.append(forecast_samples)

        # Combine batches
        if len(all_forecasts) == 1:
            combined_samples = all_forecasts[0]
        else:
            combined_samples = np.concatenate(all_forecasts, axis=0)

        # Convert to ForecastResult
        return self._to_forecast_result(
            combined_samples, dataset, horizon, quantiles
        )

    def _to_chronos_format(
        self,
        dataset: TSDataset,
    ) -> list:
        """Convert TSDataset to Chronos tensor format.

        Args:
            dataset: Input dataset

        Returns:
            List of torch tensors, one per series
        """
        import torch

        series_list = []
        for uid in dataset.series_ids:
            series_df = dataset.get_series(uid)
            values = series_df["y"].values.astype(np.float32)

            # Handle NaN values
            if np.any(np.isnan(values)):
                values = self._handle_missing_values(values)

            series_list.append(torch.tensor(values, dtype=torch.float32))

        return series_list

    def _handle_missing_values(self, values: np.ndarray) -> np.ndarray:
        """Handle missing values in series.

        Chronos requires non-NaN inputs. This fills NaNs with
        linear interpolation.

        Args:
            values: Array that may contain NaNs

        Returns:
            Array with NaNs filled
        """
        import pandas as pd

        s = pd.Series(values)
        s = s.interpolate(method="linear", limit_direction="both")
        return s.values.astype(np.float32)

    def _to_forecast_result(
        self,
        samples: np.ndarray,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None,
    ) -> ForecastResult:
        """Convert Chronos samples to ForecastResult.

        Args:
            samples: Array of shape (n_series, n_samples, horizon)
            dataset: Original dataset
            horizon: Forecast horizon
            quantiles: Quantile levels to compute

        Returns:
            ForecastResult with predictions
        """
        from tsagentkit.contracts import ForecastResult

        # Compute quantiles from samples
        if quantiles:
            quantile_values = {
                q: np.quantile(samples, q, axis=1)  # (n_series, horizon)
                for q in quantiles
            }
        else:
            # Use median as point forecast
            quantile_values = {0.5: np.median(samples, axis=1)}

        # Build result DataFrame
        result_rows = []
        for i, uid in enumerate(dataset.series_ids):
            last_date = dataset.get_series(uid)["ds"].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(1, unit=dataset.freq),
                periods=horizon,
                freq=dataset.freq,
            )

            for h in range(horizon):
                row = {
                    "unique_id": uid,
                    "ds": future_dates[h],
                    "yhat": float(quantile_values[0.5][i, h]),
                }
                # Add quantile columns
                for q in quantiles or []:
                    row[quantile_col_name(q)] = float(quantile_values[q][i, h])

                result_rows.append(row)

        result_df = pd.DataFrame(result_rows)

        # Create provenance
        provenance = self._create_provenance(dataset, horizon, quantiles)

        return ForecastResult(
            df=result_df,
            provenance=provenance,
            model_name=f"chronos-{self.config.model_size}",
            horizon=horizon,
        )

    def get_model_signature(self) -> str:
        """Return model signature for provenance.

        Returns:
            Unique signature string
        """
        return f"chronos-{self.config.model_size}-{self._device}"

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check if Chronos dependencies are installed.

        Raises:
            ImportError: If chronos-forecasting is not installed
        """
        try:
            import chronos  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting is required. "
                "Install with: pip install chronos-forecasting"
            ) from e
