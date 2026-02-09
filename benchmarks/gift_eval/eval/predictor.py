"""GluonTS predictor adapter backed by tsagentkit.run_forecast()."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import tqdm
from gluonts.dataset import Dataset as GluonTSDataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation

from tsagentkit import TaskSpec, run_forecast
from tsagentkit.utils import parse_quantile_column

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TSAgentKitPredictor(RepresentablePredictor):
    """GIFT-Eval-compatible predictor that wraps tsagentkit."""

    def __init__(
        self,
        h: int | None = None,
        freq: str | None = None,
        quantiles: list[float] | None = None,
        max_length: int | None = None,
        batch_size: int = 512,
        mode: str = "standard",
    ):
        init_prediction_length = h if h is not None else 1
        # GluonTS changed Predictor ctor signatures across releases.
        try:
            super().__init__(prediction_length=init_prediction_length, freq=freq)
        except TypeError:
            super().__init__(prediction_length=init_prediction_length)
        self.h = h
        self.freq = freq
        self.quantiles = quantiles or QUANTILES
        self.max_length = max_length
        self.imputation = LastValueImputation()
        self.batch_size = batch_size
        self.mode = mode
        self.alias = f"TSAgentKit-{mode}"

    def _to_df(self, dataset: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """Convert a GluonTS batch into tsagentkit panel DataFrame."""
        dfs: list[pd.DataFrame] = []
        metadata: dict[str, dict[str, Any]] = {}

        for entry in dataset:
            target = np.asarray(entry["target"], dtype=np.float32)

            if self.max_length and len(target) > self.max_length:
                entry["start"] += len(target) - self.max_length
                target = target[-self.max_length:]

            if np.isnan(target).any():
                target = self.imputation(target)

            if target.ndim > 1:
                raise ValueError("Only univariate time series supported")

            fcst_start = forecast_start(entry)
            uid = f"{entry['item_id']}-{fcst_start}"
            ds = pd.date_range(
                start=entry["start"].to_timestamp(),
                freq=entry["start"].freq,
                periods=len(target),
            )
            dfs.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": target}))
            metadata[uid] = {"item_id": entry["item_id"], "fcst_start": fcst_start}

        df = pd.concat(dfs, ignore_index=True).sort_values(["unique_id", "ds"])
        return df.reset_index(drop=True), metadata

    @staticmethod
    def _quantile_map(df: pd.DataFrame) -> dict[float, str]:
        mapping: dict[float, str] = {}
        for col in df.columns:
            q = parse_quantile_column(col)
            if q is not None:
                mapping[q] = col
        return mapping

    def _predict_batch(self, batch: list[dict[str, Any]], h: int, freq: str | None) -> list[Forecast]:
        """Forecast a batch through tsagentkit and adapt to QuantileForecast."""
        panel_df, metadata = self._to_df(batch)

        task_spec = TaskSpec(
            h=h,
            freq=freq,
            tsfm_policy={"mode": "preferred"},
            quantiles=self.quantiles,
        )
        run_result = run_forecast(data=panel_df, task_spec=task_spec, mode=self.mode)
        forecast_df = run_result.forecast.df

        outputs: list[Forecast] = []
        for uid, meta in metadata.items():
            fcst_uid = forecast_df[forecast_df["unique_id"] == uid]
            if fcst_uid.empty:
                arr = np.zeros((h, len(self.quantiles) + 1), dtype=np.float32)
            else:
                q_map = self._quantile_map(fcst_uid)
                mean_values = fcst_uid["yhat"].to_numpy(dtype=np.float32)
                cols = [mean_values]
                for q in self.quantiles:
                    q_col = q_map.get(float(q))
                    cols.append(
                        fcst_uid[q_col].to_numpy(dtype=np.float32) if q_col else mean_values
                    )
                arr = np.column_stack(cols)

            outputs.append(
                QuantileForecast(
                    forecast_arrays=arr.T,
                    forecast_keys=["mean"] + [f"{q}" for q in self.quantiles],
                    item_id=meta["item_id"],
                    start_date=meta["fcst_start"],
                )
            )

        return outputs

    def predict(self, dataset: GluonTSDataset, **kwargs: Any) -> list[Forecast]:
        """Generate GluonTS forecasts for all entries."""
        _ = kwargs
        forecasts: list[Forecast] = []
        batch: list[dict[str, Any]] = []

        h = self.h
        if hasattr(dataset, "test_data") and hasattr(dataset.test_data, "prediction_length"):
            h = dataset.test_data.prediction_length
        elif hasattr(dataset, "prediction_length"):
            h = dataset.prediction_length
        if h is None:
            raise ValueError("horizon `h` must be provided")

        for entry in tqdm.tqdm(dataset, total=len(dataset)):
            self.freq = entry.get("freq", self.freq)
            batch.append(entry)
            if len(batch) == self.batch_size:
                forecasts.extend(self._predict_batch(batch, h, self.freq))
                batch = []

        if batch:
            forecasts.extend(self._predict_batch(batch, h, self.freq))

        return forecasts
