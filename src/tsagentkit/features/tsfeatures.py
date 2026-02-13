"""tsfeatures adapter for statistical feature extraction."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from tsagentkit.features.config import FeatureConfig, FeatureMatrix, compute_feature_hash
from tsagentkit.features.engine import (
    create_observed_covariate_features,
    extract_panel,
    prepare_panel,
)


def _import_tsfeatures():
    try:
        import tsfeatures  # type: ignore
    except ImportError as e:
        raise ImportError(
            "tsfeatures>=0.4.5 is required for statistical feature extraction."
        ) from e

    return tsfeatures


def _resolve_feature_fns(tsfeatures_mod: Any, names: list[str]) -> list[Callable] | None:
    if not names:
        return None
    fns: list[Callable] = []
    for name in names:
        fn = getattr(tsfeatures_mod, name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown tsfeatures function: {name}")
        fns.append(fn)
    return fns


def _resolve_tsfeatures_freq(dataset: Any, config: FeatureConfig) -> int | None:
    if config.tsfeatures_freq is not None:
        return config.tsfeatures_freq
    task_spec = getattr(dataset, "task_spec", None)
    if task_spec is None:
        return None
    return task_spec.season_length


def _prefix_if_conflict(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    reserved = {"unique_id", "ds", "y"}
    conflicts = [col for col in feature_cols if col in reserved]
    if not conflicts:
        return df, feature_cols

    rename_map = {col: f"tsf_{col}" for col in conflicts}
    df = df.rename(columns=rename_map)
    updated = [rename_map.get(col, col) for col in feature_cols]
    return df, updated


def build_tsfeatures_matrix(
    dataset: Any,
    config: FeatureConfig,
    reference_time: datetime | None = None,
) -> FeatureMatrix:
    df = prepare_panel(extract_panel(dataset), reference_time)

    tsfeatures_mod = _import_tsfeatures()
    fns = _resolve_feature_fns(tsfeatures_mod, config.tsfeatures_features)

    freq = _resolve_tsfeatures_freq(dataset, config)
    dict_freqs = config.tsfeatures_dict_freqs or None

    features_df = tsfeatures_mod.tsfeatures(
        df,
        freq=freq,
        features=fns,
        dict_freqs=dict_freqs,
    )

    if "unique_id" not in features_df.columns:
        if features_df.index.name == "unique_id":
            features_df = features_df.reset_index()
        else:
            raise ValueError("tsfeatures output must include unique_id")

    feature_cols = [c for c in features_df.columns if c != "unique_id"]
    features_df, feature_cols = _prefix_if_conflict(features_df, feature_cols)

    merged = df.merge(features_df, on="unique_id", how="left")

    if config.observed_covariates:
        merged = create_observed_covariate_features(merged, config.observed_covariates)
        for col in config.observed_covariates:
            lag_col = f"{col}_lag_1"
            if lag_col in merged.columns:
                feature_cols.append(lag_col)

    if config.known_covariates:
        for col in config.known_covariates:
            if col in merged.columns and col not in feature_cols:
                feature_cols.append(col)

    if config.include_intercept:
        merged["intercept"] = 1.0
        feature_cols.append("intercept")

    config_hash = compute_feature_hash(config)

    return FeatureMatrix(
        data=merged,
        config_hash=config_hash,
        target_col="y",
        feature_cols=feature_cols,
        known_covariates=config.known_covariates,
        observed_covariates=config.observed_covariates,
    )
