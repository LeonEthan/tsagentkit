"""Native (hand-rolled) feature engineering backend.

This module preserves the original feature-engineering logic but is treated
as a non-default backend for Phase 2+.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from tsagentkit.features.matrix import FeatureMatrix
from tsagentkit.features.versioning import FeatureConfig, compute_feature_hash


def extract_panel(dataset: Any) -> pd.DataFrame:
    if hasattr(dataset, "df"):
        return dataset.df.copy()
    if hasattr(dataset, "data"):
        return dataset.data.copy()
    return dataset.copy()


def prepare_panel(df: pd.DataFrame, reference_time: datetime | None) -> pd.DataFrame:
    required = ["unique_id", "ds", "y"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    if reference_time is None:
        reference_time = df["ds"].max()

    df = df[df["ds"] <= reference_time].copy()
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def create_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"y_lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
    return df


def create_calendar_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ds = pd.to_datetime(df["ds"])
    feature_map = {
        "dayofweek": lambda d: d.dt.dayofweek,
        "month": lambda d: d.dt.month,
        "quarter": lambda d: d.dt.quarter,
        "year": lambda d: d.dt.year,
        "dayofmonth": lambda d: d.dt.day,
        "dayofyear": lambda d: d.dt.dayofyear,
        "weekofyear": lambda d: d.dt.isocalendar().week,
        "hour": lambda d: d.dt.hour,
        "minute": lambda d: d.dt.minute,
        "is_month_start": lambda d: d.dt.is_month_start.astype(int),
        "is_month_end": lambda d: d.dt.is_month_end.astype(int),
        "is_quarter_start": lambda d: d.dt.is_quarter_start.astype(int),
        "is_quarter_end": lambda d: d.dt.is_quarter_end.astype(int),
    }

    for feature in features:
        if feature in feature_map:
            df[feature] = feature_map[feature](ds)

    return df


def create_rolling_features(df: pd.DataFrame, windows: dict[int, list[str]]) -> pd.DataFrame:
    for window, aggs in windows.items():
        for agg in aggs:
            if agg == "mean":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            elif agg == "std":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .std()
                )
            elif agg == "min":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .min()
                )
            elif agg == "max":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .max()
                )
            elif agg == "sum":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .sum()
                )
            elif agg == "median":
                series = df.groupby("unique_id")["y"].transform(
                    lambda x, window=window: x.shift(1)
                    .rolling(window=window, min_periods=1)
                    .median()
                )
            else:
                continue
            df[f"y_rolling_{agg}_{window}"] = series

    return df


def create_observed_covariate_features(
    df: pd.DataFrame,
    observed_covariates: list[str],
) -> pd.DataFrame:
    for col in observed_covariates:
        if col not in df.columns:
            continue
        lag_col = f"{col}_lag_1"
        df[lag_col] = df.groupby("unique_id")[col].shift(1)
    return df


def build_native_feature_matrix(
    dataset: Any,
    config: FeatureConfig,
    reference_time: datetime | None = None,
) -> FeatureMatrix:
    df = prepare_panel(extract_panel(dataset), reference_time)

    feature_cols: list[str] = []

    if config.lags:
        df = create_lag_features(df, config.lags)
        feature_cols.extend([f"y_lag_{lag}" for lag in config.lags])

    if config.calendar_features:
        df = create_calendar_features(df, config.calendar_features)
        feature_cols.extend(config.calendar_features)

    if config.rolling_windows:
        df = create_rolling_features(df, config.rolling_windows)
        for window, aggs in config.rolling_windows.items():
            for agg in aggs:
                feature_cols.append(f"y_rolling_{agg}_{window}")

    if config.observed_covariates:
        df = create_observed_covariate_features(df, config.observed_covariates)
        for col in config.observed_covariates:
            lag_col = f"{col}_lag_1"
            if lag_col in df.columns:
                feature_cols.append(lag_col)

    if config.known_covariates:
        for col in config.known_covariates:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)

    if config.include_intercept:
        df["intercept"] = 1.0
        feature_cols.append("intercept")

    config_hash = compute_feature_hash(config)

    return FeatureMatrix(
        data=df,
        config_hash=config_hash,
        target_col="y",
        feature_cols=feature_cols,
        known_covariates=config.known_covariates,
        observed_covariates=config.observed_covariates,
    )
