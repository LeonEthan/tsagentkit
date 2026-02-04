"""Google TimesFM TSFM adapter.

Adapter for Google's TimesFM (Time Series Foundation Model).
TimesFM is a pretrained decoder-only model for time series forecasting.

Reference: https://github.com/google-research/timesfm
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


class TimesFMAdapter(TSFMAdapter):
    """Adapter for Google TimesFM foundation model.

    TimesFM is a decoder-only foundation model for time series forecasting
    that achieves strong zero-shot performance on various datasets.

    Available model sizes:
        - base: 200M parameters (default)
        - large: 500M+ parameters

    Example:
        >>> config = AdapterConfig(model_name="timesfm", model_size="base")
        >>> adapter = TimesFMAdapter(config)
        >>> adapter.load_model()
        >>> result = adapter.predict(dataset, horizon=30)

    Reference:
        https://github.com/google-research/timesfm
    """

    # Model checkpoint identifiers
    MODEL_SIZES = {
        "base": "google/timesfm-1.0-200m",
        "large": "google/timesfm-1.0-500m",
    }

    # Model configuration by size
    MODEL_CONFIGS = {
        "base": {
            "context_len": 512,
            "horizon_len": 128,
            "input_patch_len": 32,
            "output_patch_len": 128,
            "num_layers": 20,
            "model_dims": 1280,
        },
        "large": {
            "context_len": 512,
            "horizon_len": 128,
            "input_patch_len": 32,
            "output_patch_len": 128,
            "num_layers": 30,
            "model_dims": 2048,
        },
    }

    def load_model(self) -> None:
        """Load TimesFM model from checkpoint.

        Downloads and caches the model if not already present.

        Raises:
            ImportError: If timesfm is not installed
            RuntimeError: If model loading fails
        """
        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "timesfm is required for TimesFMAdapter. "
                "Install with: pip install timesfm"
            ) from e

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["base"]
        )

        config = self.MODEL_CONFIGS.get(
            self.config.model_size,
            self.MODEL_CONFIGS["base"]
        )

        try:
            self._model = timesfm.TimesFm(
                context_len=config["context_len"],
                horizon_len=config["horizon_len"],
                input_patch_len=config["input_patch_len"],
                output_patch_len=config["output_patch_len"],
                num_layers=config["num_layers"],
                model_dims=config["model_dims"],
                backend=self._device,
            )
            self._model.load_from_checkpoint(repo_id=model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM model: {e}") from e

    def fit(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ModelArtifact:
        """Prepare TimesFM for prediction.

        TimesFM is a zero-shot model and doesn't require training.
        This method validates compatibility and returns a ModelArtifact.

        Args:
            dataset: Dataset to validate
            prediction_length: Forecast horizon
            quantiles: Optional quantile levels

        Returns:
            ModelArtifact with model reference

        Raises:
            ValueError: If prediction_length exceeds model horizon
        """
        from tsagentkit.contracts import ModelArtifact

        if not self.is_loaded:
            self.load_model()

        # Validate dataset
        self._validate_dataset(dataset)

        # Check horizon against model limits
        config = self.MODEL_CONFIGS.get(
            self.config.model_size,
            self.MODEL_CONFIGS["base"]
        )
        max_horizon = config["horizon_len"]

        if prediction_length > max_horizon:
            raise ValueError(
                f"Prediction length {prediction_length} exceeds maximum "
                f"horizon {max_horizon} for {self.config.model_size} model"
            )

        return ModelArtifact(
            model=self._model,
            model_name=f"timesfm-{self.config.model_size}",
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
        """Generate forecasts using TimesFM.

        Args:
            dataset: Historical data for context
            horizon: Number of steps to forecast
            quantiles: Quantile levels for probabilistic forecasts

        Returns:
            ForecastResult with predictions and provenance
        """
        if not self.is_loaded:
            self.load_model()

        # Convert to TimesFM format
        inputs, freq = self._to_timesfm_format(dataset)

        # Generate forecasts
        # TimesFM forecast returns point forecasts and optionally quantiles
        point_forecasts, quantile_forecasts = self._model.forecast(
            inputs=inputs,
            freq=freq,
            prediction_length=horizon,
            quantile_levels=quantiles if quantiles else None,
        )

        # Convert to ForecastResult
        return self._to_forecast_result(
            point_forecasts,
            quantile_forecasts,
            dataset,
            horizon,
            quantiles,
        )

    def _to_timesfm_format(
        self,
        dataset: TSDataset,
    ) -> tuple[list, str]:
        """Convert TSDataset to TimesFM format.

        Args:
            dataset: Input dataset

        Returns:
            Tuple of (inputs list, frequency string)
        """
        inputs = []
        for uid in dataset.series_ids:
            series_df = dataset.get_series(uid)
            values = series_df["y"].values.astype(np.float32)

            # Handle NaN values
            if np.any(np.isnan(values)):
                values = self._handle_missing_values(values)

            inputs.append(values)

        # Map frequency
        freq = self._map_frequency(dataset.freq)

        return inputs, freq

    def _map_frequency(self, freq: str) -> str:
        """Map pandas frequency to TimesFM frequency format.

        TimesFM uses specific frequency codes:
        - Y: yearly
        - Q: quarterly
        - M: monthly
        - W: weekly
        - D: daily
        - H: hourly
        - T: minutely

        Args:
            freq: Pandas frequency string

        Returns:
            TimesFM frequency code
        """
        freq_map = {
            "Y": "Y", "A": "Y",       # Yearly/Annual
            "Q": "Q",                 # Quarterly
            "M": "M",                 # Monthly
            "W": "W",                 # Weekly
            "D": "D",                 # Daily
            "H": "H",                 # Hourly
            "T": "T", "min": "T",    # Minutely
            "S": "T",                 # Second -> treated as minute
        }

        # Extract base frequency code
        base_freq = freq[0].upper() if freq else "D"

        return freq_map.get(base_freq, "D")

    def _to_forecast_result(
        self,
        point_forecasts: np.ndarray,
        quantile_forecasts: dict[float, np.ndarray] | None,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None,
    ) -> ForecastResult:
        """Convert TimesFM output to ForecastResult.

        Args:
            point_forecasts: Point predictions (n_series, horizon)
            quantile_forecasts: Optional quantile predictions
            dataset: Original dataset
            horizon: Forecast horizon
            quantiles: Quantile levels

        Returns:
            ForecastResult with predictions
        """
        from tsagentkit.contracts import ForecastResult

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
                    "yhat": float(point_forecasts[i, h]),
                }

                # Add quantile columns if available
                if quantile_forecasts and quantiles:
                    for q in quantiles:
                        row[quantile_col_name(q)] = float(
                            quantile_forecasts[q][i, h]
                        )

                result_rows.append(row)

        result_df = pd.DataFrame(result_rows)
        result_df["model"] = f"timesfm-{self.config.model_size}"
        provenance = self._create_provenance(dataset, horizon, quantiles)

        return ForecastResult(
            df=result_df,
            provenance=provenance,
            model_name=f"timesfm-{self.config.model_size}",
            horizon=horizon,
        )

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
        return f"timesfm-{self.config.model_size}-{self._device}"

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check if TimesFM dependencies are installed.

        Raises:
            ImportError: If timesfm is not installed
        """
        try:
            import timesfm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "timesfm is required. "
                "Install with: pip install timesfm"
            ) from e
