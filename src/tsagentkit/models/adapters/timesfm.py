"""Google TimesFM TSFM adapter.

Adapter for Google's TimesFM (Time Series Foundation Model).
TimesFM is a pretrained decoder-only model for time series forecasting.

Reference: https://github.com/google-research/timesfm
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsagentkit.time import normalize_pandas_freq
from tsagentkit.utils import quantile_col_name

from .base import TSFMAdapter

if TYPE_CHECKING:
    from tsagentkit.contracts import ForecastResult, ModelArtifact
    from tsagentkit.series import TSDataset


class TimesFMAdapter(TSFMAdapter):
    """Adapter for Google TimesFM foundation model (2.5).

    TimesFM is a decoder-only foundation model with 200M parameters
    that achieves strong zero-shot performance on various datasets.

    Example:
        >>> adapter = TimesFMAdapter(AdapterConfig(model_name="timesfm"))
        >>> adapter.load_model()
        >>> result = adapter.predict(dataset, horizon=30)

    Reference:
        https://github.com/google-research/timesfm
    """

    # TimesFM 2.5 model checkpoint (200M parameters)
    MODEL_ID = "google/timesfm-2.5-200m-pytorch"

    # Model configuration constants
    MAX_CONTEXT = 128  # Maximum supported context length
    MAX_HORIZON = 256  # Maximum supported forecast horizon
    MIN_INPUT_LENGTH = 97  # Minimum input to avoid NaN (MAX_CONTEXT - 32 + 1)

    # Supported quantiles from the model
    SUPPORTED_QUANTILES = [round(q / 10, 1) for q in range(1, 10)]

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

        self._compiled_max_context = 0
        self._compiled_max_horizon = 0

        # Load TimesFM 2.5 model using the new API
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.MODEL_ID,
            cache_dir=self.config.cache_dir,
        )

        # Compile with default config
        self._ensure_compiled(self.MAX_CONTEXT, self.MAX_HORIZON)

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

        max_context, max_horizon = self._get_compilation_targets(
            dataset, prediction_length
        )
        self._ensure_compiled(max_context, max_horizon)

        return ModelArtifact(
            model=self._model,
            model_name="timesfm-2.5",
            config={
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

        max_context, max_horizon = self._get_compilation_targets(dataset, horizon)
        self._ensure_compiled(max_context, max_horizon)

        inputs, _freq = self._to_timesfm_format(dataset)

        # TimesFM 2.5 forecast API: forecast(horizon, inputs)
        point_forecasts, quantile_forecasts = self._model.forecast(
            horizon=horizon,
            inputs=inputs,
        )
        if point_forecasts.shape[1] > horizon:
            point_forecasts = point_forecasts[:, :horizon]
            quantile_forecasts = quantile_forecasts[:, :horizon, :]

        # Handle potential NaN in outputs (see: https://github.com/google-research/timesfm/issues/321)
        if np.any(np.isnan(point_forecasts)):
            point_forecasts = self._handle_nan_forecasts(point_forecasts, inputs)
        if quantile_forecasts is not None and np.any(np.isnan(quantile_forecasts)):
            quantile_forecasts = self._handle_nan_quantiles(quantile_forecasts, point_forecasts)

        # Convert to ForecastResult
        return self._to_forecast_result(
            point_forecasts,
            quantile_forecasts,
            dataset,
            horizon,
            quantiles,
        )

    def _handle_nan_forecasts(
        self,
        forecasts: np.ndarray,
        inputs: list[np.ndarray],
    ) -> np.ndarray:
        """Replace NaN forecasts with last valid values.

        Args:
            forecasts: Forecast array that may contain NaN
            inputs: Original input values for each series

        Returns:
            Forecast array with NaN replaced
        """
        result = forecasts.copy()
        for i in range(result.shape[0]):
            if np.any(np.isnan(result[i])):
                last_value = inputs[i][-1] if len(inputs[i]) > 0 else 0.0
                result[i] = np.nan_to_num(result[i], nan=last_value)
        return result

    def _handle_nan_quantiles(
        self,
        quantiles: np.ndarray,
        point_forecasts: np.ndarray,
    ) -> np.ndarray:
        """Replace NaN quantile forecasts.

        Args:
            quantiles: Quantile forecast array that may contain NaN
            point_forecasts: Point forecasts for fallback

        Returns:
            Quantile array with NaN replaced
        """
        result = quantiles.copy()
        for i in range(result.shape[0]):
            for h in range(result.shape[1]):
                if np.any(np.isnan(result[i, h])):
                    point_val = point_forecasts[i, h]
                    result[i, h] = np.nan_to_num(result[i, h], nan=point_val)
        return result

    def _to_timesfm_format(
        self,
        dataset: TSDataset,
    ) -> tuple[list[np.ndarray], None]:
        """Convert TSDataset to TimesFM format.

        Args:
            dataset: Input dataset

        Returns:
            Tuple of (list of input arrays, None)
            TimesFM 2.5 does not require frequency mapping.
        """
        inputs = []
        for uid in dataset.series_ids:
            series_df = dataset.get_series(uid)
            values = series_df["y"].values.astype(np.float32)

            # Handle NaN values
            if np.any(np.isnan(values)):
                values = self._handle_missing_values(values)

            # Pad short inputs to avoid NaN from attention mask issue
            # See: https://github.com/google-research/timesfm/issues/321
            if len(values) < self.MIN_INPUT_LENGTH:
                values = self._pad_input(values, self.MIN_INPUT_LENGTH)

            inputs.append(values)

        # TimesFM 2.5 does not require frequency mapping
        return inputs, None

    def _pad_input(self, values: np.ndarray, min_length: int) -> np.ndarray:
        """Pad input values to minimum length.

        Uses linear extrapolation based on the last values to avoid
        introducing artificial patterns.

        Args:
            values: Original input values
            min_length: Minimum required length

        Returns:
            Padded array
        """
        if len(values) >= min_length:
            return values

        # Calculate trend from last values for extrapolation
        n_last = min(5, len(values))
        last_values = values[-n_last:]
        trend = np.diff(last_values).mean() if n_last > 1 else 0
        last_value = values[-1]

        # Generate padded values following the trend
        n_pad = min_length - len(values)
        padded = np.arange(1, n_pad + 1) * trend + last_value

        return np.concatenate([values, padded]).astype(np.float32)

    def _to_forecast_result(
        self,
        point_forecasts: np.ndarray,
        quantile_forecasts: np.ndarray | None,
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
        freq = normalize_pandas_freq(dataset.freq)
        offset = pd.tseries.frequencies.to_offset(freq)

        quantile_values: dict[float, np.ndarray] = {}
        if quantiles:
            if quantile_forecasts is None:
                quantile_values = dict.fromkeys(quantiles, point_forecasts)
            else:
                supported = getattr(self, "_model_quantiles", self.SUPPORTED_QUANTILES)
                for q in quantiles:
                    nearest = min(supported, key=lambda v: abs(v - q))
                    idx = supported.index(nearest) + 1
                    quantile_values[q] = quantile_forecasts[:, :, idx]

        for i, uid in enumerate(dataset.series_ids):
            last_date = dataset.get_series(uid)["ds"].max()
            future_dates = pd.date_range(
                start=last_date + offset,
                periods=horizon,
                freq=freq,
            )

            for h in range(horizon):
                row = {
                    "unique_id": uid,
                    "ds": future_dates[h],
                    "yhat": float(point_forecasts[i, h]),
                }

                # Add quantile columns if available
                if quantiles:
                    for q in quantiles:
                        row[quantile_col_name(q)] = float(
                            quantile_values[q][i, h]
                        )

                result_rows.append(row)

        result_df = pd.DataFrame(result_rows)
        result_df["model"] = "timesfm-2.5"
        provenance = self._create_provenance(dataset, horizon, quantiles)

        return ForecastResult(
            df=result_df,
            provenance=provenance,
            model_name="timesfm-2.5",
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

    def _get_compilation_targets(
        self,
        dataset: TSDataset,
        horizon: int,
    ) -> tuple[int, int]:
        """Get compilation targets for model.

        Args:
            dataset: Input dataset
            horizon: Forecast horizon

        Returns:
            Tuple of (max_context, max_horizon)
        """
        max_series_len = int(
            dataset.df.groupby("unique_id").size().max()
        ) if not dataset.df.empty else 0
        target_context = self.config.max_context_length or max_series_len or self.MAX_CONTEXT
        max_context = max(self.MAX_CONTEXT, target_context)
        max_context = min(max_context, self.MAX_CONTEXT)
        max_horizon = max(self.MAX_HORIZON, horizon)
        return max_context, max_horizon

    def _ensure_compiled(self, max_context: int, max_horizon: int) -> None:
        """Ensure model is compiled with appropriate context/horizon settings.

        TimesFM 2.5 requires calling compile() with ForecastConfig before forecast().

        Args:
            max_context: Maximum context length
            max_horizon: Maximum forecast horizon
        """
        if (
            getattr(self, "_compiled_max_context", 0) >= max_context
            and getattr(self, "_compiled_max_horizon", 0) >= max_horizon
            and self._model is not None
        ):
            return

        import timesfm

        config = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
        self._model.compile(config)
        self._model_quantiles = list(self.SUPPORTED_QUANTILES)
        self._compiled_max_context = max_context
        self._compiled_max_horizon = max_horizon

    def get_model_signature(self) -> str:
        """Return model signature for provenance.

        Returns:
            Unique signature string
        """
        return f"timesfm-2.5-{self._device}"

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
