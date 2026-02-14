"""Amazon Chronos2 TSFM adapter.

Adapter for Amazon's Chronos2 time series forecasting models.
Chronos2 is a family of pretrained models based on T5 architecture
that supports zero-shot forecasting.

Reference: https://github.com/amazon-science/chronos-forecasting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.utils import quantile_col_name

from .base import TSFMAdapter, _timed_model_load

if TYPE_CHECKING:
    from tsagentkit.contracts import ForecastResult
    from tsagentkit.series import TSDataset


class ChronosAdapter(TSFMAdapter):
    """Adapter for Amazon Chronos2 time series models.

    Chronos2 is a family of pretrained time series forecasting models
    based on T5 architecture. It supports zero-shot forecasting on
    unseen time series data.

    Available model sizes:
        - small: AutoGluon Chronos-2-Small (fast, lightweight)
        - base: Amazon Chronos-2 (best accuracy)

    Example:
        >>> config = AdapterConfig(model_name="chronos", model_size="small")
        >>> adapter = ChronosAdapter(config)
        >>> adapter.load_model()
        >>> result = adapter.predict(dataset, horizon=30)

    Reference:
        https://github.com/amazon-science/chronos-forecasting
    """

    # HuggingFace model IDs for Chronos2
    MODEL_SIZES = {
        "small": "autogluon/chronos-2-small",
        "base": "amazon/chronos-2",
    }

    @_timed_model_load
    def load_model(self) -> None:
        """Load Chronos model from HuggingFace.

        Downloads and caches the model if not already present.

        Raises:
            ImportError: If chronos-forecasting is not installed
            RuntimeError: If model loading fails
        """
        try:
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting>=2.0.0 is required for ChronosAdapter. "
                "Install with: pip install 'chronos-forecasting>=2.0.0'"
            ) from e

        model_id = self.MODEL_SIZES.get(
            self.config.model_size,
            self.MODEL_SIZES["small"]
        )

        try:
            self._model = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=self._device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos model: {e}") from e

    def _prepare_model(
        self,
        dataset: TSDataset,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """Chronos requires no additional preparation (zero-shot)."""
        return {}

    def _get_model_name(self) -> str:
        """Return model name for Chronos."""
        return f"chronos-{self.config.model_size}"

    def predict(
        self,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts using Chronos.

        Supports covariate-informed forecasting when dataset contains
        past_covariates (past_x) and/or future_covariates (future_x).

        Args:
            dataset: Historical data for context, optionally with covariates
            horizon: Number of steps to forecast
            quantiles: Quantile levels for probabilistic forecasts

        Returns:
            ForecastResult with predictions and provenance
        """
        self._require_loaded("predict")
        assert self._model is not None  # type narrowing after _require_loaded

        context_df, future_df = self._to_chronos_df(dataset, horizon)

        # Use predict_df for pandas-friendly API
        # quantile_levels must not be None for Chronos 2.0
        quantile_levels = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        pred_df = self._model.predict_df(
            context_df,
            future_df=future_df if future_df is not None else None,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=horizon,
            quantile_levels=quantile_levels,
        )

        return self._to_forecast_result(pred_df, dataset, horizon, quantiles)

    def _to_chronos_df(
        self, dataset: TSDataset, horizon: int
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Convert TSDataset to Chronos DataFrame format with covariates.

        Args:
            dataset: Input dataset with optional covariates
            horizon: Forecast horizon for generating future timestamps

        Returns:
            Tuple of (context_df, future_df):
            - context_df: Historical data with target and covariates
            - future_df: Future covariates only (no target), or None if no future covariates
        """
        # Start with base columns
        df = dataset.df[["unique_id", "ds", "y"]].copy()
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Handle missing values in target
        if df["y"].isna().any():
            df["y"] = df.groupby("unique_id")["y"].transform(
                self._handle_missing_values
            )

        # Merge past covariates if available
        if dataset.past_x is not None:
            df = self._merge_covariates(df, dataset.past_x, "past")

        # Limit context length if specified
        if self.config.max_context_length:
            df = df.groupby("unique_id", as_index=False).tail(
                self.config.max_context_length
            )

        # Rename columns for Chronos format
        context_df = df.rename(
            columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"}
        )

        # Prepare future_df if future covariates are available
        future_df = None
        if dataset.future_x is not None:
            future_df = self._prepare_future_df(dataset, horizon)

        return context_df, future_df

    def _merge_covariates(
        self, df: pd.DataFrame, covariates: pd.DataFrame, cov_type: str
    ) -> pd.DataFrame:
        """Merge covariates into main DataFrame.

        Args:
            df: Main DataFrame with unique_id, ds, y
            covariates: Covariate DataFrame
            cov_type: Type of covariate ("past" or "future")

        Returns:
            Merged DataFrame
        """
        # Ensure covariates have proper index columns
        cov_df = covariates.copy()

        # Reset index if needed to get unique_id and ds as columns
        if isinstance(cov_df.index, pd.MultiIndex):
            cov_df = cov_df.reset_index()

        # Ensure required columns exist
        if "unique_id" not in cov_df.columns and "id" in cov_df.columns:
            cov_df = cov_df.rename(columns={"id": "unique_id"})
        if "ds" not in cov_df.columns and "timestamp" in cov_df.columns:
            cov_df = cov_df.rename(columns={"timestamp": "ds"})

        # Merge on unique_id and ds
        merge_cols = ["unique_id", "ds"]
        available_cols = [c for c in merge_cols if c in cov_df.columns]

        if len(available_cols) == 2:
            # Handle missing values in covariates before merging
            cov_cols = [c for c in cov_df.columns if c not in merge_cols]
            for col in cov_cols:
                if cov_df[col].isna().any():
                    cov_df[col] = self._handle_missing_values(cov_df[col])

            df = df.merge(cov_df, on=merge_cols, how="left")

        return df

    def _prepare_future_df(
        self, dataset: TSDataset, horizon: int
    ) -> pd.DataFrame | None:
        """Prepare future covariates DataFrame for Chronos.

        Args:
            dataset: Input dataset with future covariates
            horizon: Forecast horizon

        Returns:
            Future covariates DataFrame without target column
        """
        if dataset.future_x is None:
            return None

        future_df = dataset.future_x.copy()

        # Reset index if needed
        if isinstance(future_df.index, pd.MultiIndex):
            future_df = future_df.reset_index()

        # Rename columns to Chronos format
        if "unique_id" not in future_df.columns and "id" in future_df.columns:
            future_df = future_df.rename(columns={"id": "unique_id"})
        if "ds" not in future_df.columns and "timestamp" in future_df.columns:
            future_df = future_df.rename(columns={"timestamp": "ds"})

        # Rename to Chronos expected column names
        future_df = future_df.rename(
            columns={"unique_id": "item_id", "ds": "timestamp"}
        )

        # Ensure timestamp column exists - generate if needed
        if "timestamp" not in future_df.columns and dataset.future_index is not None:
            future_df = future_df.reset_index()
            if "ds" in future_df.columns:
                future_df = future_df.rename(columns={"ds": "timestamp"})

        # Handle missing values in future covariates
        for col in future_df.columns:
            if col not in ["item_id", "timestamp"] and future_df[col].isna().any():
                future_df[col] = self._handle_missing_values(future_df[col])

        return future_df

    def _to_forecast_result(
        self,
        pred_df: pd.DataFrame,
        dataset: TSDataset,
        horizon: int,
        quantiles: list[float] | None,
    ) -> ForecastResult:
        """Convert Chronos predictions to ForecastResult.

        Args:
            pred_df: DataFrame from Chronos predict_df
            dataset: Original dataset
            horizon: Forecast horizon
            quantiles: Quantile levels

        Returns:
            ForecastResult with predictions
        """
        from tsagentkit.contracts import ForecastResult

        # Map column names
        result_df = pred_df.rename(
            columns={
                "item_id": "unique_id",
                "timestamp": "ds",
                "predictions": "yhat",
            }
        )

        # Handle quantile columns (Chronos returns them as strings like "0.1", "0.5")
        if quantiles:
            quantile_cols = {}
            for col in result_df.columns:
                if col not in ["unique_id", "ds", "yhat"]:
                    try:
                        q_val = float(col)
                        if 0 < q_val < 1:
                            quantile_cols[q_val] = col
                    except (TypeError, ValueError):
                        continue

            for q in quantiles:
                if quantile_cols:
                    nearest = min(quantile_cols, key=lambda v: abs(v - q))
                    result_df[quantile_col_name(q)] = result_df[quantile_cols[nearest]]

        # Select and order columns
        keep_cols = ["unique_id", "ds", "yhat"]
        for q in quantiles or []:
            col = quantile_col_name(q)
            if col in result_df.columns:
                keep_cols.append(col)

        result_df = result_df[keep_cols].copy()
        result_df["model"] = f"chronos-{self.config.model_size}"

        provenance = self._create_provenance(dataset, horizon, quantiles)

        return ForecastResult(
            df=result_df,
            provenance=provenance,
            model_name=f"chronos-{self.config.model_size}",
            horizon=horizon,
        )

    def get_model_signature(self) -> str:
        """Return model signature for provenance."""
        return f"chronos-{self.config.model_size}-{self._device}"

    @classmethod
    def _check_dependencies_impl(cls) -> None:
        """Check if Chronos dependencies are installed."""
        try:
            import chronos  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting is required. "
                "Install with: pip install chronos-forecasting"
            ) from e

    @classmethod
    def _get_capability_spec(cls, adapter_name: str) -> dict[str, Any]:
        return {
            "adapter_name": adapter_name,
            "provider": "amazon",
            "is_zero_shot": True,
            "supports_quantiles": True,
            "supports_past_covariates": True,
            "supports_future_covariates": True,
            "supports_static_covariates": False,
            "max_context_length": None,
            "max_horizon": None,
            "dependencies": ["torch", "chronos-forecasting>=2.0.0"],
            "notes": "Chronos 2 pipeline supports context + future covariates via predict_df.",
        }
