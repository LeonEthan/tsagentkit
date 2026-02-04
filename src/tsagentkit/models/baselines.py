"""Baseline model implementations backed by statsforecast."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tsagentkit.contracts import ModelArtifact
from tsagentkit.utils import normalize_quantile_columns, quantile_col_name


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _get_statsforecast_models() -> dict[str, type]:
    try:
        from statsforecast import models as sf_models
    except ImportError as exc:
        raise ImportError(
            "statsforecast is required for baseline models. "
            "Install with: uv sync (or pip install statsforecast)"
        ) from exc

    model_map: dict[str, type] = {}
    for key, attr in {
        "naive": "Naive",
        "seasonalnaive": "SeasonalNaive",
        "historicaverage": "HistoricAverage",
        "theta": "Theta",
        "windowaverage": "WindowAverage",
        "movingaverage": "WindowAverage",
        "seasonalwindowaverage": "SeasonalWindowAverage",
        "autoets": "AutoETS",
        "ets": "AutoETS",
    }.items():
        model_cls = getattr(sf_models, attr, None)
        if model_cls is not None:
            model_map[key] = model_cls

    croston_cls = None
    for name in ("CrostonClassic", "CrostonOptimized", "CrostonSBA", "CrostonTSB"):
        croston_cls = getattr(sf_models, name, None)
        if croston_cls is not None:
            break
    if croston_cls is not None:
        model_map["croston"] = croston_cls

    return model_map


def is_baseline_model(model_name: str) -> bool:
    """Return True if model_name is a supported baseline."""
    normalized = _normalize_name(model_name)
    return normalized in _get_statsforecast_models()


def _build_model(
    model_name: str,
    config: dict[str, Any],
) -> tuple[Any, str]:
    model_map = _get_statsforecast_models()
    normalized = _normalize_name(model_name)

    if normalized not in model_map:
        raise ValueError(f"Unknown baseline model: {model_name}")

    model_cls = model_map[normalized]
    model_key = model_cls.__name__

    kwargs: dict[str, Any] = {}
    if normalized == "seasonalnaive":
        kwargs["season_length"] = int(config.get("season_length", 1))
    elif normalized in {"windowaverage", "movingaverage"}:
        kwargs["window_size"] = int(config.get("window_size", 3))
    elif normalized == "seasonalwindowaverage":
        kwargs["season_length"] = int(config.get("season_length", 1))
        kwargs["window_size"] = int(config.get("window_size", 3))
    elif normalized in {"autoets", "ets"}:
        kwargs["season_length"] = int(config.get("season_length", 1))
    return model_cls(**kwargs), model_key


def fit_baseline(
    model_name: str,
    dataset: Any,
    config: dict[str, Any],
) -> ModelArtifact:
    """Fit a baseline model using statsforecast."""
    from statsforecast import StatsForecast

    model, model_key = _build_model(model_name, config)

    try:
        sf = StatsForecast(
            models=[model],
            freq=dataset.task_spec.freq,
            n_jobs=1,
        )
        sf.fit(dataset.df)
    except TypeError:
        sf = StatsForecast(
            df=dataset.df,
            models=[model],
            freq=dataset.task_spec.freq,
            n_jobs=1,
        )
        sf.fit()

    return ModelArtifact(
        model=sf,
        model_name=model_name,
        config=config,
        metadata={
            "baseline_model": model_key,
            "freq": dataset.task_spec.freq,
        },
    )


def _extract_point_column(forecast_df: pd.DataFrame) -> str:
    value_cols = [c for c in forecast_df.columns if c not in {"unique_id", "ds"}]
    point_cols = [c for c in value_cols if "lo-" not in c and "hi-" not in c]
    if not point_cols:
        raise ValueError("No point forecast column found in statsforecast output.")
    if len(point_cols) == 1:
        return point_cols[0]
    # Prefer plain yhat if present
    if "yhat" in point_cols:
        return "yhat"
    return point_cols[0]


def _level_for_quantile(q: float) -> int:
    return int(round((1 - 2 * abs(q - 0.5)) * 100))


def _find_interval_column(
    forecast_df: pd.DataFrame,
    level: int,
    kind: str,
) -> str | None:
    suffix = f"{kind}-{level}"
    matches = [c for c in forecast_df.columns if c.endswith(suffix)]
    if matches:
        return matches[0]
    return None


def predict_baseline(
    model_artifact: ModelArtifact,
    dataset: Any | None,
    horizon: int,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate baseline forecasts using statsforecast."""
    sf = model_artifact.model
    levels: list[int] = []

    if quantiles:
        for q in quantiles:
            if q == 0.5:
                continue
            levels.append(_level_for_quantile(q))
        levels = sorted({lvl for lvl in levels if lvl > 0})

    try:
        forecast_df = sf.forecast(h=horizon, level=levels or None)
    except TypeError:
        if dataset is None:
            raise
        forecast_df = sf.forecast(df=dataset.df, h=horizon, level=levels or None)
    if "unique_id" not in forecast_df.columns or "ds" not in forecast_df.columns:
        forecast_df = forecast_df.reset_index()

    point_col = _extract_point_column(forecast_df)
    if point_col != "yhat":
        forecast_df = forecast_df.rename(columns={point_col: "yhat"})

    # Normalize quantile columns
    if quantiles:
        for q in quantiles:
            col_name = quantile_col_name(q)
            if q == 0.5:
                forecast_df[col_name] = forecast_df["yhat"]
                continue
            level = _level_for_quantile(q)
            kind = "lo" if q < 0.5 else "hi"
            interval_col = _find_interval_column(forecast_df, level, kind)
            if interval_col is None:
                forecast_df[col_name] = forecast_df["yhat"]
            else:
                forecast_df[col_name] = forecast_df[interval_col]

    forecast_df = normalize_quantile_columns(forecast_df)

    # Keep standard column order
    if "model" not in forecast_df.columns:
        forecast_df["model"] = model_artifact.model_name
    cols = [
        "unique_id",
        "ds",
        "model",
        "yhat",
    ] + [c for c in forecast_df.columns if c.startswith("q")]
    return forecast_df[cols]
