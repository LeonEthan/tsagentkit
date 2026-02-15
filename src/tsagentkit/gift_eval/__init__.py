"""GIFT-Eval dataset loading utilities for tsagentkit.

This module provides thin wrappers around GIFT-Eval Hugging Face datasets
for use with tsagentkit's core API (run_forecast, forecast).

Example:
    from tsagentkit.gift_eval import Dataset, Term
    from tsagentkit import run_forecast, TaskSpec

    dataset = Dataset(name="m4_yearly", term=Term.SHORT, storage_path="./data")

    # Convert training data to DataFrame for tsagentkit
    train_data = dataset.training_dataset
    df = gluonts_to_dataframe(train_data, dataset.freq)

    # Use tsagentkit's API directly
    spec = TaskSpec(h=dataset.prediction_length, freq=dataset.freq)
    result = run_forecast(df, spec)
"""

from .data import (
    Dataset,
    GIFTEvalDataset,
    Term,
    download_data,
    MultivariateToUnivariate,
    normalize_freq,
    itemize_start,
)

__all__ = [
    "Dataset",
    "GIFTEvalDataset",
    "Term",
    "download_data",
    "MultivariateToUnivariate",
    "normalize_freq",
    "itemize_start",
]
