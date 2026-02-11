"""GIFT-Eval dataset loading utilities."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator
from enum import Enum
from functools import cached_property
from pathlib import Path

import datasets
import pyarrow.compute as pc
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}


def itemize_start(data_entry: DataEntry) -> DataEntry:
    """Convert numpy scalar timestamp start into a Python scalar."""
    data_entry["start"] = data_entry["start"].item()
    return data_entry


def normalize_freq(freq: str) -> str:
    """Normalize pandas frequency aliases to GluonTS/GIFT-Eval conventions."""
    mapping = {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "s": "S"}
    return mapping.get(freq, freq)


class MultivariateToUnivariate(Transformation):
    """Convert a multivariate dataset into one univariate entry per dimension."""

    def __init__(self, field: str):
        self.field = field

    def __call__(self, data_it: Iterable[DataEntry], is_train: bool = False) -> Iterator[DataEntry]:
        _ = is_train  # kept for transformation protocol compatibility
        for entry in data_it:
            item_id = entry["item_id"]
            for idx, val in enumerate(entry[self.field]):
                new_entry = entry.copy()
                new_entry[self.field] = val
                new_entry["item_id"] = f"{item_id}_dim{idx}"
                yield new_entry


class Term(Enum):
    """Evaluation term multiplier."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        return {Term.SHORT: 1, Term.MEDIUM: 10, Term.LONG: 15}[self]


class GIFTEvalDataset:
    """Thin wrapper around GIFT-Eval Hugging Face datasets with GluonTS splits."""

    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
        storage_path: Path | str | None = None,
        storage_env_var: str = "GIFT_EVAL",
    ):
        if storage_path is None:
            storage_root = Path(os.getenv(storage_env_var, "."))
        else:
            storage_root = Path(storage_path)

        self.hf_dataset = datasets.load_from_disk(str(storage_root / name)).with_format("numpy")
        self.term = Term(term)
        self.name = name

        process = ProcessDataEntry(self.freq, one_dim_target=self.target_dim == 1)
        gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            gluonts_dataset = MultivariateToUnivariate("target").apply(gluonts_dataset)
        self.gluonts_dataset = gluonts_dataset

    @cached_property
    def prediction_length(self) -> int:
        """Prediction length derived from dataset family and term multiplier."""
        freq = normalize_freq(norm_freq_str(to_offset(self.freq).name))
        pred_len = M4_PRED_LENGTH[freq] if "m4" in self.name else PRED_LENGTH[freq]
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        """Dataset frequency alias."""
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        """Target dimensionality (1 for univariate)."""
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if len(target.shape) > 1 else 1

    @cached_property
    def windows(self) -> int:
        """Number of rolling windows used by GIFT-Eval protocol."""
        if "m4" in self.name:
            return 1
        windows = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, windows), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        col = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(col, 0, 1)))
        else:
            lengths = pc.list_value_length(col)
        return int(min(lengths.to_numpy()))

    @property
    def training_dataset(self) -> TrainingDataset:
        """Training split used by most baseline implementations."""
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * (self.windows + 1),
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        """Validation split aligned with benchmark rolling windows."""
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        """Evaluation instances matching benchmark split protocol."""
        _, test_template = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )


SHORT_DATASETS = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]

MED_LONG_DATASETS = [
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]


def get_all_dataset_terms() -> list[tuple[str, str]]:
    """Return the canonical full matrix of (dataset, term) configurations."""
    configs: list[tuple[str, str]] = [(dataset, "short") for dataset in SHORT_DATASETS]
    for dataset in MED_LONG_DATASETS:
        configs.extend([(dataset, "medium"), (dataset, "long")])
    return configs


DATASETS_WITH_TERMS = tuple(get_all_dataset_terms())
FULL_MATRIX_SIZE = len(DATASETS_WITH_TERMS)

# Keep this alias to match upstream naming conventions (`gift_eval.data.Dataset`).
Dataset = GIFTEvalDataset
