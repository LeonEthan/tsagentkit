"""
Running GIFT-Eval with tsagentkit v2.0 (Ensemble Mode)

This script evaluates tsagentkit v2.0 on the GIFT-Eval benchmark using
the ensemble forecasting API (forecast) with TSFM + statistical models.

Usage:
    python tsagentkit_quick.py
    python tsagentkit_quick.py --debug --max-series 50
    python tsagentkit_quick.py --download --max-series 1000
    python tsagentkit_quick.py --datasets m4_yearly m4_monthly --max-series 200
    python tsagentkit_quick.py --ensemble-method mean  # Use mean instead of median
"""

# %% Import dependencies and setup
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gluonts.dataset.util import forecast_start
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
from gluonts.time_feature import get_seasonality

from tsagentkit import forecast, ForecastConfig
from tsagentkit.gift_eval import Dataset, Term, download_data, normalize_freq

if TYPE_CHECKING:
    from gluonts.dataset import DataEntry

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger("gluonts").setLevel(logging.ERROR)

# %% CLI Arguments
parser = argparse.ArgumentParser(
    description="Evaluate tsagentkit on GIFT-Eval benchmark",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s                           # Run with defaults (debug mode, 100 series)
  %(prog)s --debug --max-series 50   # Debug mode with 50 series
  %(prog)s --download                # Download data first
  %(prog)s --datasets m4_yearly      # Run specific dataset(s)
  %(prog)s --all-datasets            # Run all datasets (no debug mode)
    """,
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=True,
    help="Debug mode: use only a few short datasets (default: True)",
)
parser.add_argument(
    "--no-debug",
    action="store_false",
    dest="debug",
    help="Disable debug mode: run all datasets",
)
parser.add_argument(
    "--max-series",
    type=int,
    default=100,
    metavar="N",
    help="Limit number of series per dataset for quick testing (default: 100, 0=all)",
)
parser.add_argument(
    "--download",
    action="store_true",
    help="Download GIFT-Eval datasets if not already available",
)
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["m4_yearly"],
    metavar="DATASET",
    help="Specific dataset(s) to run (default: m4_yearly)",
)
parser.add_argument(
    "--all-datasets",
    action="store_true",
    help="Run all datasets (overrides --debug and --datasets)",
)
parser.add_argument(
    "--storage-path",
    type=Path,
    default=Path("./data/gift-eval"),
    metavar="PATH",
    help="Path to GIFT-Eval data (default: ./data/gift-eval)",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("./results/tsagentkit_quick"),
    metavar="PATH",
    help="Output directory for results (default: ./results/tsagentkit_quick)",
)
parser.add_argument(
    "--mode",
    choices=["quick", "standard", "strict"],
    default="quick",
    help="tsagentkit run mode (default: quick)",
)
parser.add_argument(
    "--ensemble-method",
    choices=["median", "mean"],
    default="median",
    help="Ensemble aggregation method (default: median)",
)
parser.add_argument(
    "--tsfm-mode",
    choices=["required", "preferred", "disabled"],
    default="preferred",
    help="TSFM policy (default: preferred)",
)

args = parser.parse_args()

# %% Configuration
# Paths
storage_path = args.storage_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Model configuration
model_name = f"tsagentkit_v2_{args.mode}_{args.ensemble_method}"
mode = args.mode
ensemble_method = args.ensemble_method
tsfm_mode = args.tsfm_mode

# Debug mode
debug = args.debug and not args.all_datasets
if args.all_datasets:
    debug = False

# Dataset selection
if args.all_datasets:
    single_dataset = None
elif args.datasets:
    single_dataset = args.datasets
else:
    single_dataset = ["m4_yearly"]

# Limit number of series (0 means no limit)
max_series = args.max_series if args.max_series > 0 else None

# Data download
download = args.download

# Dataset lists (moved from gift_eval module)
short_datasets = [
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

med_long_datasets = [
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

# Dataset properties (moved from gift_eval module)
dataset_properties = {
    "bitbrains_fast_storage": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 2},
    "bitbrains_rnd": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 2},
    "bizitobs_application": {"domain": "Web/CloudOps", "frequency": "10S", "num_variates": 2},
    "bizitobs_l2c": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 7},
    "bizitobs_service": {"domain": "Web/CloudOps", "frequency": "10S", "num_variates": 2},
    "car_parts": {"domain": "Sales", "frequency": "M", "num_variates": 1},
    "covid_deaths": {"domain": "Healthcare", "frequency": "D", "num_variates": 1},
    "electricity": {"domain": "Energy", "frequency": "W", "num_variates": 1},
    "ett1": {"domain": "Energy", "frequency": "W", "num_variates": 7},
    "ett2": {"domain": "Energy", "frequency": "W", "num_variates": 7},
    "hierarchical_sales": {"domain": "Sales", "frequency": "W-WED", "num_variates": 1},
    "hospital": {"domain": "Healthcare", "frequency": "M", "num_variates": 1},
    "jena_weather": {"domain": "Nature", "frequency": "D", "num_variates": 21},
    "kdd_cup_2018": {"domain": "Nature", "frequency": "D", "num_variates": 1},
    "loop_seattle": {"domain": "Transport", "frequency": "D", "num_variates": 1},
    "m4_daily": {"domain": "Econ/Fin", "frequency": "D", "num_variates": 1},
    "m4_hourly": {"domain": "Econ/Fin", "frequency": "H", "num_variates": 1},
    "m4_monthly": {"domain": "Econ/Fin", "frequency": "M", "num_variates": 1},
    "m4_quarterly": {"domain": "Econ/Fin", "frequency": "Q", "num_variates": 1},
    "m4_weekly": {"domain": "Econ/Fin", "frequency": "W", "num_variates": 1},
    "m4_yearly": {"domain": "Econ/Fin", "frequency": "A", "num_variates": 1},
    "m_dense": {"domain": "Transport", "frequency": "D", "num_variates": 1},
    "restaurant": {"domain": "Sales", "frequency": "D", "num_variates": 1},
    "saugeen": {"domain": "Nature", "frequency": "M", "num_variates": 1},
    "solar": {"domain": "Energy", "frequency": "W", "num_variates": 1},
    "sz_taxi": {"domain": "Transport", "frequency": "H", "num_variates": 1},
    "temperature_rain": {"domain": "Nature", "frequency": "D", "num_variates": 1},
    "us_births": {"domain": "Healthcare", "frequency": "M", "num_variates": 1},
}

# Pretty names for dataset keys
pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Combined datasets
all_datasets = sorted(set(short_datasets + med_long_datasets))

# Debug mode: use specified datasets or default to m4_yearly
if debug:
    all_datasets = single_dataset if single_dataset else ["m4_yearly"]

# Download data if requested
if download:
    download_data(storage_path=storage_path)

print(f"storage_path={storage_path.resolve()}")
print(f"output_dir={output_dir.resolve()}")
print(f"Mode: {mode}")
print(f"Ensemble method: {ensemble_method}")
print(f"TSFM mode: {tsfm_mode}")
print(f"Debug mode: {debug}")
if max_series:
    print(f"Max series per dataset: {max_series}")
print(f"Download data: {download}")
print(f"Total datasets: {len(all_datasets)}")


# %% Helper functions
def gluonts_to_dataframe(
    gluonts_dataset: list[DataEntry], freq: str, max_series: int | None = None
) -> pd.DataFrame:
    """Convert GluonTS dataset to tsagentkit panel DataFrame format.

    Args:
        gluonts_dataset: List of GluonTS data entries
        freq: Frequency string
        max_series: Maximum number of series to include (for testing)

    Returns:
        DataFrame with columns [unique_id, ds, y]
    """
    dfs = []
    series_count = 0
    for entry in gluonts_dataset:
        if max_series is not None and series_count >= max_series:
            break

        target = np.asarray(entry["target"], dtype=np.float32)

        if target.ndim > 1:
            # Multivariate - expand to univariate entries
            for i, dim_target in enumerate(target):
                ds = pd.date_range(
                    start=entry["start"].to_timestamp(),
                    freq=freq,
                    periods=len(dim_target),
                )
                uid = f"{entry['item_id']}_dim{i}"
                dfs.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": dim_target}))
            series_count += 1
        else:
            ds = pd.date_range(
                start=entry["start"].to_timestamp(),
                freq=freq,
                periods=len(target),
            )
            dfs.append(
                pd.DataFrame({"unique_id": entry["item_id"], "ds": ds, "y": target})
            )
            series_count += 1

    return pd.concat(dfs, ignore_index=True).sort_values(["unique_id", "ds"])


def get_test_actuals(test_data, prediction_length: int, max_series: int | None = None) -> pd.DataFrame:
    """Extract actual test values from GluonTS test data.

    Returns a DataFrame with columns [unique_id, ds, y] for the forecast horizon.
    """
    dfs = []
    series_count = 0
    for train_entry, test_entry in test_data:
        if max_series is not None and series_count >= max_series:
            break
        target = np.asarray(test_entry["target"], dtype=np.float32)
        fcst_start = forecast_start(test_entry)

        if target.ndim > 1:
            # Multivariate
            for i, dim_target in enumerate(target):
                ds = pd.date_range(
                    start=fcst_start.to_timestamp(),
                    freq=test_entry["start"].freq,
                    periods=prediction_length,
                )
                uid = f"{test_entry['item_id']}_dim{i}"
                dfs.append(
                    pd.DataFrame({"unique_id": uid, "ds": ds, "y": dim_target[:prediction_length]})
                )
            series_count += 1
        else:
            ds = pd.date_range(
                start=fcst_start.to_timestamp(),
                freq=test_entry["start"].freq,
                periods=prediction_length,
            )
            dfs.append(
                pd.DataFrame({
                    "unique_id": test_entry["item_id"],
                    "ds": ds,
                    "y": target[:prediction_length],
                })
            )
            series_count += 1

    return pd.concat(dfs, ignore_index=True)


def align_forecast_with_actuals(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align forecast with actuals by position within each series.

    GIFT-Eval uses a rolling window protocol where forecast dates may not match
    test dates directly. We align by position (last N values) within each series.
    """
    # Get unique series
    series_ids = forecast_df["unique_id"].unique()
    aligned_forecasts = []

    for uid in series_ids:
        fcst = forecast_df[forecast_df["unique_id"] == uid].copy()
        actual = actuals_df[actuals_df["unique_id"] == uid]

        if len(fcst) == 0 or len(actual) == 0:
            continue

        # Align by position: use the last N values where N = len(actual)
        n_actual = len(actual)
        if len(fcst) >= n_actual:
            # Take the last n_actual values from forecast
            fcst_aligned = fcst.tail(n_actual).copy()
            # Replace forecast dates with actual dates for merging
            fcst_aligned["ds"] = actual["ds"].values
            aligned_forecasts.append(fcst_aligned)

    if not aligned_forecasts:
        return pd.DataFrame()

    return pd.concat(aligned_forecasts, ignore_index=True)


def compute_metrics(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    seasonality: int,
    quantiles: list[float],
) -> dict[str, float]:
    """Compute evaluation metrics from forecast and actuals.

    Args:
        forecast_df: Forecast DataFrame with columns [unique_id, ds, yhat, quantile_0.1, ...]
        actuals_df: Actual values DataFrame with columns [unique_id, ds, y]
        seasonality: Seasonal period for MASE
        quantiles: List of quantile levels

    Returns:
        Dictionary of metric values
    """
    # Align forecast with actuals by position (GIFT-Eval rolling window protocol)
    forecast_aligned = align_forecast_with_actuals(forecast_df, actuals_df)

    # Merge forecast with actuals
    merged = forecast_aligned.merge(actuals_df, on=["unique_id", "ds"], suffixes=("", "_actual"))

    # Extract forecast and actual arrays
    forecast_values = merged["yhat"].to_numpy()
    actual_values = merged["y"].to_numpy()

    # For simplicity, compute metrics using GluonTS metric classes directly
    # We need to reshape to (num_series, prediction_length) format

    # Get unique series and prediction length per series
    grouped = merged.groupby("unique_id")
    num_series = len(grouped)
    pred_length = len(merged) // num_series if num_series > 0 else 0

    if num_series == 0 or pred_length == 0:
        # Return NaN for all metrics
        return {
            "MSE[mean]": np.nan,
            "MSE[0.5]": np.nan,
            "MAE[0.5]": np.nan,
            "MASE[0.5]": np.nan,
            "MAPE[0.5]": np.nan,
            "sMAPE[0.5]": np.nan,
            "MSIS": np.nan,
            "RMSE[mean]": np.nan,
            "NRMSE[mean]": np.nan,
            "ND[0.5]": np.nan,
            "mean_weighted_sum_quantile_loss": np.nan,
        }

    # Reshape to (num_series, pred_length)
    forecast_matrix = forecast_values.reshape(num_series, pred_length)
    actual_matrix = actual_values.reshape(num_series, pred_length)

    # Create forecast arrays with quantiles for CRPS
    # Shape: (num_series, num_quantiles + 1, pred_length) where +1 is for mean
    num_quantiles = len(quantiles)
    forecast_arrays = np.zeros((num_series, num_quantiles + 1, pred_length))
    forecast_arrays[:, 0, :] = forecast_matrix  # mean

    # Add quantile forecasts
    for i, q in enumerate(quantiles):
        q_col = f"q{q}"
        if q_col in merged.columns:
            q_values = merged[q_col].to_numpy().reshape(num_series, pred_length)
            forecast_arrays[:, i + 1, :] = q_values
        else:
            # Fall back to mean if quantile not available
            forecast_arrays[:, i + 1, :] = forecast_matrix

    # Compute metrics
    # Note: GluonTS metrics expect specific input formats
    # For simplicity, we compute basic versions here

    # MSE
    mse = np.mean((forecast_matrix - actual_matrix) ** 2)

    # MAE
    mae = np.mean(np.abs(forecast_matrix - actual_matrix))

    # RMSE
    rmse = np.sqrt(mse)

    # MAPE
    mape = np.mean(np.abs((actual_matrix - forecast_matrix) / (actual_matrix + 1e-8))) * 100

    # sMAPE
    smape = (
        np.mean(
            2
            * np.abs(forecast_matrix - actual_matrix)
            / (np.abs(actual_matrix) + np.abs(forecast_matrix) + 1e-8)
        )
        * 100
    )

    # MASE - simplified version using naive forecast
    # Compute in-sample MAE for naive forecast
    mae_naive = np.mean(np.abs(actual_matrix[:, seasonality:] - actual_matrix[:, :-seasonality]))
    mase = mae / (mae_naive + 1e-8)

    # NRMSE - normalized by mean of actuals
    nrmse = rmse / (np.mean(np.abs(actual_matrix)) + 1e-8)

    # ND - normalized deviation
    nd = np.sum(np.abs(forecast_matrix - actual_matrix)) / np.sum(np.abs(actual_matrix) + 1e-8)

    # MSIS - simplified interval score
    # Use quantiles if available, otherwise use mean
    q_low_col = f"q{quantiles[0]}"
    q_high_col = f"q{quantiles[-1]}"

    if q_low_col in merged.columns and q_high_col in merged.columns:
        q_low = merged[q_low_col].to_numpy().reshape(num_series, pred_length)
        q_high = merged[q_high_col].to_numpy().reshape(num_series, pred_length)
        alpha = quantiles[-1] - quantiles[0]

        interval_score = np.mean(
            (q_high - q_low)
            + (2 / alpha) * (q_low - actual_matrix) * (actual_matrix < q_low)
            + (2 / alpha) * (actual_matrix - q_high) * (actual_matrix > q_high)
        )
        msis = interval_score / (mae_naive + 1e-8)
    else:
        msis = np.nan

    # Mean Weighted Sum Quantile Loss (CRPS-like)
    crps = 0.0
    for q in quantiles:
        q_col = f"q{q}"
        if q_col in merged.columns:
            q_values = merged[q_col].to_numpy().reshape(num_series, pred_length)
            crps += np.mean(2 * np.abs((actual_matrix - q_values) * ((actual_matrix <= q_values).astype(float) - q)))
    crps /= len(quantiles) if quantiles else 1.0

    return {
        "MSE[mean]": mse,
        "MSE[0.5]": mse,  # Same as mean for point forecast
        "MAE[0.5]": mae,
        "MASE[0.5]": mase,
        "MAPE[0.5]": mape,
        "sMAPE[0.5]": smape,
        "MSIS": msis,
        "RMSE[mean]": rmse,
        "NRMSE[mean]": nrmse,
        "ND[0.5]": nd,
        "mean_weighted_sum_quantile_loss": crps,
    }


# %% Main evaluation loop
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

result_columns = [
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

results = []

for ds_name in all_datasets:
    print(f"Processing dataset: {ds_name}")
    terms = ["short", "medium", "long"]

    for term in terms:
        if (term == "medium" or term == "long") and ds_name not in med_long_datasets:
            continue

        # Parse dataset key and frequency
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"

        # Create dataset
        term_enum = Term(term)
        to_univariate = (
            False
            if Dataset(name=ds_name, term=term_enum, to_univariate=False, storage_path=storage_path).target_dim
            == 1
            else True
        )

        dataset = Dataset(
            name=ds_name, term=term_enum, to_univariate=to_univariate, storage_path=storage_path
        )

        # Get test data which provides (train_entry, test_entry) pairs
        # train_entry contains the history aligned with test ground truth
        test_data = list(dataset.test_data)

        # Normalize frequency for tsagentkit compatibility
        freq = normalize_freq(dataset.freq)

        # Convert train entries from test_data to DataFrame
        train_entries = [train_entry for train_entry, _ in test_data]
        train_df = gluonts_to_dataframe(train_entries, freq, max_series=max_series)

        # Skip if no data
        if len(train_df) == 0:
            print(f"  Skipping {ds_config}: empty training data")
            continue

        # Normalize frequency for tsagentkit compatibility
        freq = normalize_freq(dataset.freq)

        # Get seasonality for metrics
        seasonality = get_seasonality(freq)

        # Run forecast with tsagentkit v2.0 ensemble API
        try:
            result = forecast(
                train_df,
                h=dataset.prediction_length,
                freq=freq,
                quantiles=quantiles,
                mode=mode,
                ensemble_method=ensemble_method,
                tsfm_mode=tsfm_mode,
            )
            forecast_df = result.forecast.df
            # Log ensemble info
            ensemble_count = forecast_df["_ensemble_count"].iloc[0] if "_ensemble_count" in forecast_df.columns else "N/A"
            print(f"  Ensemble: {ensemble_count} models contributed")
        except Exception as e:
            print(f"  Error forecasting {ds_config}: {e}")
            continue

        # Get test actuals from test_data
        test_actuals_df = get_test_actuals(dataset.test_data, dataset.prediction_length, max_series=max_series)

        # Compute metrics
        metrics = compute_metrics(
            forecast_df=forecast_df,
            actuals_df=test_actuals_df,
            seasonality=seasonality,
            quantiles=quantiles,
        )

        # Build result row
        result_row = [
            ds_config,
            model_name,
            metrics["MSE[mean]"],
            metrics["MSE[0.5]"],
            metrics["MAE[0.5]"],
            metrics["MASE[0.5]"],
            metrics["MAPE[0.5]"],
            metrics["sMAPE[0.5]"],
            metrics["MSIS"],
            metrics["RMSE[mean]"],
            metrics["NRMSE[mean]"],
            metrics["ND[0.5]"],
            metrics["mean_weighted_sum_quantile_loss"],
            dataset_properties[ds_key]["domain"],
            dataset_properties[ds_key]["num_variates"],
        ]

        results.append((ds_config, result_row))
        print(
            f"  {ds_config}: MASE={metrics['MASE[0.5]']:.6f}, CRPS={metrics['mean_weighted_sum_quantile_loss']:.6f}"
        )

# %% Save results
csv_file_path = output_dir / "all_results.csv"
results_df = pd.DataFrame([row for _, row in results], columns=result_columns)
results_df.to_csv(csv_file_path, index=False)

print(f"\nResults written to {csv_file_path}")
print(f"Total results: {len(results)}")

# %% Compute normalized scores vs Seasonal Naive baseline
baseline_path = Path("./results/seasonal_naive/all_results.csv")

if baseline_path.exists():
    seasonal_naive = pd.read_csv(baseline_path).sort_values("dataset")
    df = results_df.sort_values("dataset")

    baseline_by_dataset = seasonal_naive.set_index("dataset")
    aligned = baseline_by_dataset.loc[df["dataset"]]

    df["normalized MASE"] = (
        df["eval_metrics/MASE[0.5]"].to_numpy()
        / aligned["eval_metrics/MASE[0.5]"].to_numpy()
    )
    df["normalized CRPS"] = (
        df["eval_metrics/mean_weighted_sum_quantile_loss"].to_numpy()
        / aligned["eval_metrics/mean_weighted_sum_quantile_loss"].to_numpy()
    )

    mase = float(np.exp(np.mean(np.log(df["normalized MASE"].to_numpy()))))
    crps = float(np.exp(np.mean(np.log(df["normalized CRPS"].to_numpy()))))

    print(f"Final GIFT-Eval performance of {model_name}:")
    print(f"MASE = {mase}")
    print(f"CRPS = {crps}")
else:
    print(
        "Normalized MASE/CRPS not computed "
        "(missing ./results/seasonal_naive/all_results.csv)."
    )

# %% View results
print(f"\nResults preview:")
print(results_df.head(10))
