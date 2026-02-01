"""Salesforce Moirai TSFM adapter.

Adapter for Salesforce's Moirai universal time series forecasting model.
Moirai is a transformer-based model trained on large-scale time series data.

Reference: https://github.com/SalesforceAIResearch/uni2ts
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


class MoiraiAdapter(TSFMAdapter):
    """Adapter for Salesforce Moirai foundation model.

    Moirai is a universal time series forecasting transformer trained on
    large-scale time series data. It uses a patch-based architecture for
    handling variable-length series.

    Available model sizes:
        - small: 200M parameters
        - base: 400M parameters (recommended default)
        - large: 1B+ parameters

    Example:
        >>> config = AdapterConfig(model_name="moirai", model_size="base")
        >>> adapter = MoiraiAdapter(config)
        >>> adapter.load_model()
        >>> result = adapter.predict(dataset, horizon=30)

    Reference:
        https://github.com/SalesforceAIResearch/uni2ts
    """

    # HuggingFace model IDs for each size
    MODEL_SIZES = {
        "small": "Salesforce/moirai-1.0-R-small",
        "base": "Salesforce/moirai-1.0-R-base",
        "large": "Salesforce/moirai-1.1-R-large",
    }

    # Patch sizes for different frequencies (model-specific)
    PATCH_SIZES = {
        "small": {"D": 32, "H": 24, "W": 4, "M": 1, "Q": 1, "Y": 1},
        "base": {"D": 32, "H": 24, "W": 4, "M": 1, "Q": 1, "Y": 1},
        "large": {"D": 32, "H": 24, "W": 4, "M": 1, "Q": 1, "Y": 1},
    }

    def load_model(self) -> None:
        """Load Moirai model from HuggingFace.

        Downloads and caches the model if not already present.

        Raises:
            ImportError: If uni2ts is not installed
            RuntimeError: If model loading fails
        """
        try:
            from uni2ts.model.moirai import MoiraiForecast
        except ImportError as e:
            raise ImportError(
                "uni2ts is required for MoiraiAdapter. "
                "Install with: pip install uni2ts"
            ) from e

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["base"]
        )

        try:
            self._model = MoiraiForecast.load_from_checkpoint(
                checkpoint_path=model_id,
            )
            # Move to device if supported
            if hasattr(self._model, 'to'):
                import torch
                self._model = self._model.to(torch.device(self._device))
        except Exception as e:
            raise RuntimeError(f"Failed to load Moirai model: {e}") from e

    def fit(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ModelArtifact:
        """Prepare Moirai for prediction.

        Moirai is a zero-shot model and doesn't require training.
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

        return ModelArtifact(
            model=self._model,
            model_name=f"moirai-{self.config.model_size}",
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
        """Generate forecasts using Moirai.

        Uses Moirai's patch-based architecture for variable-length series.

        Args:
            dataset: Historical data for context
            horizon: Number of steps to forecast
            quantiles: Quantile levels for probabilistic forecasts

        Returns:
            ForecastResult with predictions and provenance
        """
        if not self.is_loaded:
            self.load_model()

        # Process each series
        all_forecasts = []
        for uid in dataset.series_ids:
            series_df = dataset.get_series(uid)
            forecast = self._predict_single_series(series_df, horizon, quantiles)
            all_forecasts.append(forecast)

        # Combine into single result
        return self._combine_forecasts(
            all_forecasts, dataset, horizon, quantiles
        )

    def _predict_single_series(
        self,
        series_df: pd.DataFrame,
        horizon: int,
        quantiles: list[float] | None,
    ) -> dict:
        """Predict for a single time series.

        Args:
            series_df: DataFrame with single series data
            horizon: Forecast horizon
            quantiles: Quantile levels

        Returns:
            Dictionary with forecast data
        """
        from uni2ts.model.moirai import MoiraiForecast

        values = series_df["y"].values.astype(np.float32)

        # Handle NaN values
        if np.any(np.isnan(values)):
            values = self._handle_missing_values(values)

        # Get frequency and determine patch size
        # Default to daily if unclear
        inferred_freq = pd.infer_freq(series_df["ds"])
        freq_code = inferred_freq[0].upper() if inferred_freq else "D"
        patch_size = self._get_patch_size(freq_code)

        # Prepare sample for Moirai
        sample = {
            "target": values,
            "start": series_df["ds"].iloc[0],
        }

        # Generate forecast
        forecast = self._model(
            [sample],
            prediction_length=horizon,
        )

        # Moirai returns distributions; sample for quantiles
        if quantiles:
            # Generate multiple samples for quantile estimation
            samples = []
            for _ in range(self.config.num_samples):
                sample_forecast = self._model(
                    [sample],
                    prediction_length=horizon,
                )
                samples.append(sample_forecast[0].samples.numpy())
            forecast_samples = np.stack(samples, axis=0)
        else:
            # Use mean prediction
            forecast_samples = forecast[0].mean.numpy()

        return {
            "samples": forecast_samples,
            "last_date": series_df["ds"].max(),
        }

    def _combine_forecasts(
        self,
        forecasts: list[dict],
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None,
    ) -> ForecastResult:
        """Combine individual series forecasts into ForecastResult.

        Args:
            forecasts: List of forecast dictionaries
            dataset: Original dataset
            horizon: Forecast horizon
            quantiles: Quantile levels

        Returns:
            Combined ForecastResult
        """
        from tsagentkit.contracts import ForecastResult

        result_rows = []
        for i, (uid, forecast) in enumerate(zip(dataset.series_ids, forecasts)):
            samples = forecast["samples"]
            last_date = forecast["last_date"]

            # Compute quantiles
            if quantiles and len(samples.shape) > 1:
                quantile_values = {
                    q: np.quantile(samples, q, axis=0)
                    for q in quantiles
                }
                point_forecast = quantile_values[0.5]
            else:
                point_forecast = samples if len(samples.shape) == 1 else samples.mean(axis=0)
                quantile_values = {0.5: point_forecast}

            # Generate future dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(1, unit=dataset.freq),
                periods=horizon,
                freq=dataset.freq,
            )

            for h in range(horizon):
                row = {
                    "unique_id": uid,
                    "ds": future_dates[h],
                    "yhat": float(point_forecast[h]),
                }
                for q in quantiles or []:
                    row[quantile_col_name(q)] = float(quantile_values[q][h])

                result_rows.append(row)

        result_df = pd.DataFrame(result_rows)
        provenance = self._create_provenance(dataset, horizon, quantiles)

        return ForecastResult(
            df=result_df,
            provenance=provenance,
            model_name=f"moirai-{self.config.model_size}",
            horizon=horizon,
        )

    def _get_patch_size(self, freq: str) -> int:
        """Get appropriate patch size for frequency.

        Args:
            freq: Frequency code (D, H, W, M, Q, Y)

        Returns:
            Patch size for the model
        """
        freq_map = freq[0].upper() if freq else "D"
        size_config = self.PATCH_SIZES.get(self.config.model_size, self.PATCH_SIZES["base"])
        return size_config.get(freq_map, 32)

    def _handle_missing_values(self, values: np.ndarray) -> np.ndarray:
        """Handle missing values in series.

        Args:
            values: Array that may contain NaNs

        Returns:
            Array with NaNs filled
        """
        import pandas as pd

        s = pd.Series(values)
        s = s.interpolate(method="linear", limit_direction="both")
        return s.fillna(s.mean()).values.astype(np.float32)

    def get_model_signature(self) -> str:
        """Return model signature for provenance.

        Returns:
            Unique signature string
        """
        return f"moirai-{self.config.model_size}-{self._device}"

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check if Moirai dependencies are installed.

        Raises:
            ImportError: If uni2ts is not installed
        """
        try:
            import uni2ts  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "uni2ts is required. "
                "Install with: pip install uni2ts"
            ) from e
