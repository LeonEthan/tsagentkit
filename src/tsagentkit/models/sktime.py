"""Sktime forecaster adapter with covariate support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tsagentkit.contracts import ETaskSpecIncompatible, ForecastResult, ModelArtifact
from tsagentkit.time import make_future_index
from tsagentkit.utils import normalize_quantile_columns


@dataclass(frozen=True)
class SktimeModelBundle:
    """Container for per-series sktime forecasters."""

    model_name: str
    forecasters: dict[str, Any]
    exog_columns: list[str]
    static_columns: list[str]
    future_columns: list[str]


def _require_sktime() -> None:
    try:
        import sktime  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "sktime is required for sktime adapters. Install with: uv sync --extra features"
        ) from exc


def _make_forecaster(model_key: str, season_length: int | None) -> Any:
    _require_sktime()

    from sktime.forecasting.naive import NaiveForecaster

    if model_key in {"naive", "last"}:
        return NaiveForecaster(strategy="last")
    if model_key in {"seasonal_naive", "seasonal"}:
        sp = season_length or 1
        return NaiveForecaster(strategy="last", sp=sp)

    raise ValueError(f"Unsupported sktime model key: {model_key}")


def _get_exog_columns(
    covariates: Any | None,
    plan: Any,
) -> tuple[list[str], list[str]]:
    static_cols: list[str] = []
    future_cols: list[str] = []

    if covariates is None:
        return static_cols, future_cols

    if getattr(plan, "use_static", True) and covariates.static_x is not None:
        static_cols = [c for c in covariates.static_x.columns if c != "unique_id"]

    if getattr(plan, "use_future_known", True) and covariates.future_x is not None:
        future_cols = [
            c for c in covariates.future_x.columns if c not in {"unique_id", "ds"}
        ]

    return static_cols, future_cols


def _build_train_exog(
    dataset: Any,
    covariates: Any,
    uid: str,
    ds_index: pd.Index,
    static_cols: list[str],
    future_cols: list[str],
) -> pd.DataFrame | None:
    if not static_cols and not future_cols:
        return None

    exog = pd.DataFrame(index=ds_index)

    if future_cols:
        panel = dataset.panel_with_covariates
        if panel is None:
            raise ETaskSpecIncompatible(
                "Future-known covariates require panel_with_covariates for history.",
                context={"unique_id": uid},
            )
        hist = panel[panel["unique_id"] == uid][["ds"] + future_cols].copy()
        hist = hist.set_index("ds")
        exog = exog.join(hist[future_cols], how="left")

    if static_cols:
        static_row = covariates.static_x
        if static_row is None:
            raise ETaskSpecIncompatible(
                "Static covariates requested but none provided.",
                context={"unique_id": uid},
            )
        static_row = static_row[static_row["unique_id"] == uid]
        if static_row.empty:
            raise ETaskSpecIncompatible(
                "Static covariate row missing for series.",
                context={"unique_id": uid},
            )
        for col in static_cols:
            value = static_row.iloc[0][col]
            exog[col] = value

    if exog.isna().any().any():
        raise ETaskSpecIncompatible(
            "Missing exogenous covariate values in training data.",
            context={"unique_id": uid},
        )

    return exog


def _build_future_exog(
    covariates: Any,
    uid: str,
    static_cols: list[str],
    future_cols: list[str],
) -> pd.DataFrame | None:
    if not static_cols and not future_cols:
        return None

    if covariates.future_index is None:
        raise ETaskSpecIncompatible(
            "Future index is required for exogenous forecasting.",
            context={"unique_id": uid},
        )

    future_index = covariates.future_index
    future_index = future_index[future_index["unique_id"] == uid].copy()
    future_index = future_index.set_index("ds")

    exog = pd.DataFrame(index=future_index.index)

    if future_cols:
        future = covariates.future_x
        if future is None:
            raise ETaskSpecIncompatible(
                "Future-known covariates requested but none provided.",
                context={"unique_id": uid},
            )
        future = future[future["unique_id"] == uid][["ds"] + future_cols].copy()
        future = future.set_index("ds")
        exog = exog.join(future[future_cols], how="left")

    if static_cols:
        static_row = covariates.static_x
        if static_row is None:
            raise ETaskSpecIncompatible(
                "Static covariates requested but none provided.",
                context={"unique_id": uid},
            )
        static_row = static_row[static_row["unique_id"] == uid]
        if static_row.empty:
            raise ETaskSpecIncompatible(
                "Static covariate row missing for series.",
                context={"unique_id": uid},
            )
        for col in static_cols:
            value = static_row.iloc[0][col]
            exog[col] = value

    if exog.isna().any().any():
        raise ETaskSpecIncompatible(
            "Missing exogenous covariate values for forecast horizon.",
            context={"unique_id": uid},
        )

    return exog


def fit_sktime(
    model_name: str,
    dataset: Any,
    plan: Any,
    covariates: Any | None = None,
) -> ModelArtifact:
    """Fit per-series sktime forecasters with exogenous covariates."""
    model_key = model_name.split("sktime-", 1)[-1]
    static_cols, future_cols = _get_exog_columns(covariates, plan)
    forecasters: dict[str, Any] = {}

    for uid in dataset.series_ids:
        series_df = dataset.get_series(uid).copy()
        series_df = series_df.sort_values("ds")
        y = series_df.set_index("ds")["y"].astype(float)

        X_train = None
        if covariates is not None:
            X_train = _build_train_exog(
                dataset=dataset,
                covariates=covariates,
                uid=uid,
                ds_index=y.index,
                static_cols=static_cols,
                future_cols=future_cols,
            )

        forecaster = _make_forecaster(model_key, dataset.task_spec.season_length)
        forecaster.fit(y, X=X_train)
        forecasters[uid] = forecaster

    bundle = SktimeModelBundle(
        model_name=model_name,
        forecasters=forecasters,
        exog_columns=sorted(static_cols + future_cols),
        static_columns=static_cols,
        future_columns=future_cols,
    )

    return ModelArtifact(
        model=bundle,
        model_name=model_name,
        config={
            "model_key": model_key,
            "static_columns": static_cols,
            "future_columns": future_cols,
        },
    )


def predict_sktime(
    dataset: Any,
    artifact: ModelArtifact,
    spec: Any,
    covariates: Any | None = None,
) -> ForecastResult:
    """Predict with per-series sktime forecasters."""
    _require_sktime()

    from sktime.forecasting.base import ForecastingHorizon

    bundle: SktimeModelBundle = artifact.model
    rows: list[dict[str, Any]] = []

    if covariates is None:
        future_index = make_future_index(dataset.df, spec.horizon, spec.freq)
    else:
        future_index = covariates.future_index
        if future_index is None:
            future_index = make_future_index(dataset.df, spec.horizon, spec.freq)

    for uid, forecaster in bundle.forecasters.items():
        future_dates = future_index[future_index["unique_id"] == uid]["ds"]
        future_dates = pd.to_datetime(future_dates).sort_values()
        fh = ForecastingHorizon(pd.DatetimeIndex(future_dates), is_relative=False)

        X_future = None
        if covariates is not None and (bundle.static_columns or bundle.future_columns):
            X_future = _build_future_exog(
                covariates=covariates,
                uid=uid,
                static_cols=bundle.static_columns,
                future_cols=bundle.future_columns,
            )

        y_pred = forecaster.predict(fh, X=X_future)
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.iloc[:, 0]

        for ds, value in y_pred.items():
            rows.append(
                {
                    "unique_id": uid,
                    "ds": pd.Timestamp(ds),
                    "model": artifact.model_name,
                    "yhat": float(value),
                }
            )

    result_df = pd.DataFrame(rows)
    result_df = normalize_quantile_columns(result_df)

    provenance = _basic_provenance(dataset, spec, artifact)
    return ForecastResult(
        df=result_df,
        provenance=provenance,
        model_name=artifact.model_name,
        horizon=spec.horizon,
    )


def _basic_provenance(dataset: Any, spec: Any, artifact: ModelArtifact) -> Any:
    from datetime import datetime, timezone

    from tsagentkit.contracts import Provenance
    from tsagentkit.utils import compute_data_signature

    return Provenance(
        run_id=f"sktime_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_signature=compute_data_signature(dataset.df),
        task_signature=spec.model_hash(),
        plan_signature=artifact.signature,
        model_signature=artifact.signature,
        metadata={"adapter": "sktime"},
    )


__all__ = ["SktimeModelBundle", "fit_sktime", "predict_sktime"]
