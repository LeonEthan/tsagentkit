"""Evaluation runner for GIFT-Eval benchmark."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download

from .data_loader import GIFTEvalDataset
from .predictor import QUANTILES, TSAgentKitPredictor

logger = logging.getLogger(__name__)

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILES),
]

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


class GIFTEvalRunner:
    """Run tsagentkit on one GIFT-Eval dataset configuration."""

    @staticmethod
    def download_data(storage_path: Path | str) -> None:
        """Download the full GIFT-Eval dataset snapshot from Hugging Face."""
        logger.info("Downloading GIFT-Eval dataset to %s ...", storage_path)
        snapshot_download(
            repo_id="Salesforce/GiftEval",
            repo_type="dataset",
            local_dir=storage_path,
        )
        logger.info("Download complete")

    def __init__(
        self,
        dataset_name: str,
        term: str,
        output_path: Path | str | None = None,
        storage_path: Path | str | None = None,
        mode: str = "standard",
        preload_adapters: list[str] | None = None,
        dataset_properties_path: Path | str | None = None,
    ):
        self.dataset_properties = self._load_dataset_properties(dataset_properties_path)

        if "/" in dataset_name:
            ds_key, ds_freq = dataset_name.split("/", maxsplit=1)
            ds_key = ds_key.lower()
        else:
            ds_key = dataset_name.lower()
            ds_freq = self.dataset_properties[ds_key]["frequency"]

        pretty_key = PRETTY_NAMES.get(ds_key, ds_key)
        self.ds_key = pretty_key
        self.ds_lookup_key = pretty_key if pretty_key in self.dataset_properties else ds_key
        self.ds_config = f"{pretty_key}/{ds_freq}/{term}"
        self.dataset_name = dataset_name
        self.mode = mode
        self.preload_adapters = preload_adapters
        self.output_path = Path(output_path) if output_path else None

        temp_dataset = GIFTEvalDataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            storage_path=storage_path,
        )
        to_univariate = temp_dataset.target_dim != 1
        self.dataset = GIFTEvalDataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            storage_path=storage_path,
        )
        self.seasonality = get_seasonality(self.dataset.freq)

    @staticmethod
    def _load_dataset_properties(dataset_properties_path: Path | str | None) -> dict[str, dict[str, object]]:
        if dataset_properties_path is None:
            dataset_properties_path = Path(__file__).with_name("dataset_properties.json")
        path = Path(dataset_properties_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("dataset_properties.json must contain a top-level object")
        return payload

    def evaluate(
        self,
        predictor: TSAgentKitPredictor | None = None,
        batch_size: int = 512,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Evaluate one dataset/term pair and append to all_results.csv."""
        created_predictor = False
        if predictor is None:
            predictor = TSAgentKitPredictor(
                mode=self.mode,
                batch_size=batch_size,
                preload_adapters=self.preload_adapters,
            )
            created_predictor = True

        logger.info(
            "Evaluating %s/%s with TSAgentKit (%s mode)",
            self.dataset_name,
            self.dataset.term.value,
            self.mode,
        )

        try:
            res = evaluate_model(
                predictor,
                test_data=self.dataset.test_data,
                metrics=METRICS,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=self.seasonality,
            )
        finally:
            if created_predictor:
                predictor.close()

        results_data = [
            [
                self.ds_config,
                f"TSAgentKit-{self.mode}",
                res["MSE[mean]"].iloc[0],
                res["MSE[0.5]"].iloc[0],
                res["MAE[0.5]"].iloc[0],
                res["MASE[0.5]"].iloc[0],
                res["MAPE[0.5]"].iloc[0],
                res["sMAPE[0.5]"].iloc[0],
                res["MSIS"].iloc[0],
                res["RMSE[mean]"].iloc[0],
                res["NRMSE[mean]"].iloc[0],
                res["ND[0.5]"].iloc[0],
                res["mean_weighted_sum_quantile_loss"].iloc[0],
                self.dataset_properties[self.ds_lookup_key]["domain"],
                self.dataset_properties[self.ds_lookup_key]["num_variates"],
            ]
        ]

        columns = [
            "dataset",
            "model",
            "eval_metrics/MSE[mean]",
            "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[0.5]",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]",
            "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS",
            "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain",
            "num_variates",
        ]
        current_df = pd.DataFrame(results_data, columns=columns)

        if self.output_path:
            csv_file = self.output_path / "all_results.csv"
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            if csv_file.exists() and not overwrite:
                current_df = pd.concat([pd.read_csv(csv_file), current_df], ignore_index=True)
            current_df.to_csv(csv_file, index=False)
            logger.info("Results saved: %s", csv_file)

        return current_df
