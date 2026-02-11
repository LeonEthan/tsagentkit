"""GIFT-Eval orchestration helpers aligned with official notebook usage."""

from __future__ import annotations

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

from . import data as gift_data
from .dataset_properties import DATASET_PROPERTIES
from .predictor import QUANTILES, TSAgentKitPredictor

logger = logging.getLogger(__name__)

GIFTEvalDataset = gift_data.GIFTEvalDataset
DATASETS_WITH_TERMS = gift_data.DATASETS_WITH_TERMS
FULL_MATRIX_SIZE = gift_data.FULL_MATRIX_SIZE
SHORT_DATASETS = gift_data.SHORT_DATASETS
MED_LONG_DATASETS = gift_data.MED_LONG_DATASETS

RESULT_COLUMNS = [
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


class GIFTEval:
    """Evaluate one `(dataset_name, term)` configuration and write leaderboard-style rows."""

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
        dataset_properties: dict[str, dict[str, object]] | None = None,
    ):
        self.dataset_properties = DATASET_PROPERTIES if dataset_properties is None else dataset_properties

        if "/" in dataset_name:
            ds_key, ds_freq = dataset_name.split("/", maxsplit=1)
            ds_key = ds_key.lower()
        else:
            ds_key = dataset_name.lower()
            ds_freq = str(self.dataset_properties[ds_key]["frequency"])

        pretty_key = PRETTY_NAMES.get(ds_key, ds_key)
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

    def evaluate_predictor(
        self,
        predictor,
        batch_size: int = 512,
        overwrite: bool = False,
        model_name: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate a GluonTS-compatible predictor and append rows to all_results.csv."""
        logger.info(
            "Evaluating %s/%s with mode=%s",
            self.dataset_name,
            self.dataset.term.value,
            self.mode,
        )

        res = evaluate_model(
            predictor,
            test_data=self.dataset.test_data,
            metrics=METRICS,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=self.seasonality,
        )

        row = [
            self.ds_config,
            model_name or str(getattr(predictor, "alias", f"tsagentkit-{self.mode}")),
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
        current_df = pd.DataFrame([row], columns=RESULT_COLUMNS)

        if self.output_path:
            csv_file = self.output_path / "all_results.csv"
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            if csv_file.exists() and not overwrite:
                current_df = pd.concat([pd.read_csv(csv_file), current_df], ignore_index=True)
            current_df.to_csv(csv_file, index=False)
            logger.info("Results saved: %s", csv_file)

        return current_df

    def evaluate(
        self,
        predictor: TSAgentKitPredictor | None = None,
        batch_size: int = 512,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Backward-compatible evaluate method.

        If `predictor` is omitted, create one `TSAgentKitPredictor` for this call.
        """
        created_predictor = False
        if predictor is None:
            predictor = TSAgentKitPredictor(
                mode=self.mode,
                batch_size=batch_size,
                preload_adapters=self.preload_adapters,
            )
            created_predictor = True

        try:
            return self.evaluate_predictor(predictor, batch_size=batch_size, overwrite=overwrite)
        finally:
            if created_predictor:
                predictor.close()


def run_combinations(
    combinations: list[tuple[str, str]],
    *,
    storage_path: Path | str,
    output_path: Path | str,
    mode: str = "standard",
    batch_size: int = 512,
    preload_adapters: list[str] | None = None,
    overwrite_results: bool = False,
    predictor: TSAgentKitPredictor | None = None,
) -> Path:
    """Run multiple configurations while reusing one predictor instance."""
    storage = Path(storage_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    results_csv = output / "all_results.csv"
    if overwrite_results and results_csv.exists():
        results_csv.unlink()

    created_predictor = False
    if predictor is None:
        predictor = TSAgentKitPredictor(
            mode=mode,
            batch_size=batch_size,
            preload_adapters=preload_adapters,
        )
        created_predictor = True

    try:
        for idx, (dataset_name, term) in enumerate(combinations, start=1):
            logger.info("[%d/%d] %s/%s", idx, len(combinations), dataset_name, term)
            runner = GIFTEval(
                dataset_name=dataset_name,
                term=term,
                output_path=output,
                storage_path=storage,
                mode=mode,
                preload_adapters=preload_adapters,
            )
            row_df = runner.evaluate_predictor(
                predictor=predictor,
                batch_size=batch_size,
                overwrite=False,
            )
            row = row_df.iloc[-1]
            logger.info(
                "MASE=%.4f | sMAPE=%.4f | CRPS=%.4f",
                float(row["eval_metrics/MASE[0.5]"]),
                float(row["eval_metrics/sMAPE[0.5]"]),
                float(row["eval_metrics/mean_weighted_sum_quantile_loss"]),
            )
    finally:
        if created_predictor and predictor is not None:
            predictor.close()

    return results_csv


# Backward-compatible alias used by local scripts/tests.
GIFTEvalRunner = GIFTEval
