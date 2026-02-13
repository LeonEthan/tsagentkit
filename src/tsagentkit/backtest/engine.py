"""Rolling window backtest engine.

Implements expanding and sliding window backtesting with strict
temporal integrity (no random splits allowed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from tsagentkit.contracts import CVFrame
from tsagentkit.eval import evaluate_forecasts
from tsagentkit.utils import call_with_optional_kwargs, drop_future_rows, normalize_quantile_columns

from .aggregation import (
    aggregate_metrics as compute_aggregate_metrics,
)
from .aggregation import (
    build_series_metrics,
    compute_segment_metrics,
    compute_temporal_metrics,
    series_metrics_from_frame,
    summary_to_metrics,
)
from .report import (
    BacktestReport,
    SegmentMetrics,
    SeriesMetrics,
    TemporalMetrics,
    WindowResult,
)
from .splitting import (
    cross_validation_split as _cross_validation_split,
)
from .splitting import (
    generate_cutoffs,
    validate_temporal_ordering,
)
from .window_utils import build_window_covariates, reconcile_forecast

if TYPE_CHECKING:
    from tsagentkit.contracts import TaskSpec
    from tsagentkit.router import PlanSpec
    from tsagentkit.series import TSDataset


def rolling_backtest(
    dataset: TSDataset,
    spec: TaskSpec,
    plan: PlanSpec,
    fit_func: Callable[[TSDataset, PlanSpec], Any] | None = None,
    predict_func: Callable[[TSDataset, Any, TaskSpec], Any] | None = None,  # Also accepts covariates: Optional kwarg
    n_windows: int = 5,
    window_strategy: Literal["expanding", "sliding"] = "expanding",
    min_train_size: int | None = None,
    step_size: int | None = None,
    reconcile: bool = True,
    route_decision: Any | None = None,
) -> BacktestReport:
    """Execute rolling window backtest.

    Performs temporal cross-validation using expanding or sliding windows.
    Random splits are strictly forbidden.

    For hierarchical datasets, applies forecast reconciliation to ensure
    coherence across the hierarchy (enabled by default).

    Args:
        dataset: TSDataset with time series data
        spec: Task specification
        plan: Execution plan with model configuration
        fit_func: Function to fit model: fit_func(train_dataset, plan) (defaults to models.fit)
        predict_func: Function to predict: predict_func(train_dataset, model_artifact, spec, covariates=None)
            (defaults to models.predict). Must accept covariates as optional kwarg.
        n_windows: Number of backtest windows (default: 5)
        window_strategy: "expanding" or "sliding" (default: "expanding")
        min_train_size: Minimum training observations per series
        step_size: Step size between windows (default: spec.horizon)
        reconcile: Whether to reconcile forecasts for hierarchical data (default: True)
        route_decision: Optional RouteDecision for including routing info in metadata (v1.0)

    Returns:
        BacktestReport with results from all windows

    Raises:
        ESplitRandomForbidden: If random splitting is detected
        EBacktestInsufficientData: If not enough data for requested windows
    """
    # Resolve default model functions
    if fit_func is None or predict_func is None:
        from tsagentkit.models import fit as default_fit
        from tsagentkit.models import predict as default_predict

        fit_func = default_fit if fit_func is None else fit_func
        predict_func = default_predict if predict_func is None else predict_func

    # Drop future rows (y is null beyond last observed per series) before validation
    df, _ = drop_future_rows(dataset.df)

    # Validate temporal ordering (guardrail)
    validate_temporal_ordering(df)

    # Set defaults
    horizon = spec.horizon
    step = step_size if step_size is not None else horizon
    season_length = spec.season_length or 1

    if min_train_size is None:
        # Default: at least 2 seasons worth of data
        min_train_size = max(season_length * 2, 10)

    # Get date range
    all_dates = pd.to_datetime(df["ds"].unique())
    all_dates = sorted(all_dates)

    min_required = min_train_size + (n_windows - 1) * step + horizon
    if len(all_dates) < min_required:
        from tsagentkit.contracts import EBacktestInsufficientData

        raise EBacktestInsufficientData(
            f"Insufficient data for {n_windows} windows. "
            f"Have {len(all_dates)} dates, need at least {min_required}",
            context={
                "n_dates": len(all_dates),
                "min_required": min_required,
                "n_windows_requested": n_windows,
            },
        )

    # Generate windows
    window_results: list[WindowResult] = []
    series_metrics_agg: dict[str, list[dict]] = {}
    cv_frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    # Calculate window cutoffs
    cutoffs = generate_cutoffs(
        all_dates,
        n_windows=n_windows,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        strategy=window_strategy,
    )

    for window_idx, (cutoff_date, test_dates) in enumerate(cutoffs):
        try:
            # Split data
            train_df = df[df["ds"] < cutoff_date].copy()
            test_df = df[df["ds"].isin(test_dates)].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                errors.append(
                    {
                        "window": window_idx,
                        "error": "Empty train or test set",
                        "cutoff": str(cutoff_date),
                    }
                )
                continue

            from tsagentkit.series import TSDataset

            train_ds = TSDataset.from_dataframe(train_df, spec, validate=False)
            if dataset.is_hierarchical() and dataset.hierarchy:
                train_ds = train_ds.with_hierarchy(dataset.hierarchy)
            window_covariates = None
            try:
                window_covariates = build_window_covariates(
                    dataset=dataset,
                    task_spec=spec,
                    cutoff_date=pd.Timestamp(cutoff_date),
                    panel_for_index=train_df,
                )
            except (ValueError, TypeError, KeyError) as e:
                errors.append(
                    {
                        "window": window_idx,
                        "stage": "covariate_alignment",
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )
                if not plan.allow_drop_covariates:
                    raise
            if window_covariates is not None:
                train_ds = train_ds.with_covariates(
                    window_covariates,
                    panel_with_covariates=dataset.panel_with_covariates,
                    covariate_bundle=dataset.covariate_bundle,
                )

            # Fit model (with fallback handled by fit_func)
            try:
                model = call_with_optional_kwargs(
                    fit_func,
                    train_ds,
                    plan,
                    covariates=window_covariates,
                )
            except (RuntimeError, ValueError, MemoryError) as e:
                errors.append(
                    {
                        "window": window_idx,
                        "stage": "fit",
                        "error": str(e),
                        "type": type(e).__name__,
                        "model": plan.candidate_models[0] if plan.candidate_models else None,
                    }
                )
                continue

            model_name = getattr(model, "model_name", None)
            if model_name is None and hasattr(model, "metadata"):
                model_name = model.metadata.get("model_name") if model.metadata else None
            if model_name is None:
                model_name = plan.candidate_models[0] if plan.candidate_models else "model"

            # Predict using training context
            try:
                predictions = call_with_optional_kwargs(
                    predict_func,
                    train_ds,
                    model,
                    spec,
                    covariates=window_covariates,
                )
            except (RuntimeError, ValueError, MemoryError) as e:
                errors.append(
                    {
                        "window": window_idx,
                        "stage": "predict",
                        "error": str(e),
                        "type": type(e).__name__,
                        "model": model_name,
                    }
                )
                continue
            if isinstance(predictions, dict):
                raise ValueError("predict_func must return DataFrame or ForecastResult")
            if hasattr(predictions, "df"):
                predictions = predictions.df

            # Align predictions to test dates only
            predictions = predictions.merge(
                test_df[["unique_id", "ds"]],
                on=["unique_id", "ds"],
                how="inner",
            )
            predictions["model"] = model_name

            cv_frame = predictions.merge(
                test_df[["unique_id", "ds", "y"]],
                on=["unique_id", "ds"],
                how="left",
            )
            cv_frame["cutoff"] = pd.Timestamp(cutoff_date)
            cv_frames.append(cv_frame)

            # Apply reconciliation if hierarchical
            if reconcile and dataset.is_hierarchical() and dataset.hierarchy:
                predictions = reconcile_forecast(
                    predictions,
                    dataset.hierarchy,
                    "bottom_up",
                )
            predictions = normalize_quantile_columns(predictions)
            predictions["model"] = model_name

            # Compute window + series metrics via eval utilities
            merged_metrics = predictions.merge(
                test_df[["unique_id", "ds", "y"]],
                on=["unique_id", "ds"],
                how="left",
            )
            metric_frame, summary = evaluate_forecasts(
                merged_metrics,
                train_df=train_df,
                season_length=season_length,
                id_col="unique_id",
                ds_col="ds",
                target_col="y",
                model_col="model",
                pred_col="yhat",
                cutoff_col=None,
            )

            window_metrics = summary_to_metrics(summary.df, model_name)
            series_window_metrics = series_metrics_from_frame(metric_frame.df, model_name)
            for uid, metrics in series_window_metrics.items():
                if not metrics:
                    continue
                if uid not in series_metrics_agg:
                    series_metrics_agg[uid] = []
                series_metrics_agg[uid].append(metrics)

            # Create window result
            window_result = WindowResult(
                window_index=window_idx,
                train_start=str(train_df["ds"].min()),
                train_end=str(train_df["ds"].max()),
                test_start=str(test_df["ds"].min()),
                test_end=str(test_df["ds"].max()),
                metrics=window_metrics,
                num_series=test_df["unique_id"].nunique(),
                num_observations=len(test_df),
            )
            window_results.append(window_result)

        except Exception as e:
            errors.append(
                {
                    "window": window_idx,
                    "error": str(e),
                    "type": type(e).__name__,
                }
            )

    # Aggregate metrics across all windows
    aggregate_metrics = compute_aggregate_metrics(series_metrics_agg)
    series_metrics = build_series_metrics(series_metrics_agg)

    # Compute segment metrics (by sparsity class) if sparsity profile available
    segment_metrics = compute_segment_metrics(series_metrics, dataset)

    # Compute temporal metrics if datetime information available
    temporal_metrics = compute_temporal_metrics(series_metrics_agg, df)

    return BacktestReport(
        n_windows=len(window_results),
        strategy=window_strategy,
        window_results=window_results,
        aggregate_metrics=aggregate_metrics,
        series_metrics=series_metrics,
        segment_metrics=segment_metrics,
        temporal_metrics=temporal_metrics,
        errors=errors,
        metadata={
            "horizon": horizon,
            "step_size": step,
            "min_train_size": min_train_size,
            "primary_model": plan.candidate_models[0] if plan.candidate_models else None,
            "decision_summary": {
                "plan_name": getattr(plan, "plan_name", None),
                "primary_model": plan.candidate_models[0] if plan.candidate_models else None,
                "reasons": route_decision.reasons if route_decision else ["rule_based_router"],
                "buckets": route_decision.buckets if route_decision else [],
                "stats": route_decision.stats if route_decision else {},
            },
        },
        cv_frame=CVFrame(df=pd.concat(cv_frames, ignore_index=True)) if cv_frames else None,
    )


def _summary_to_metrics(summary_df: pd.DataFrame, model_name: str | None) -> dict[str, float]:
    return summary_to_metrics(summary_df, model_name)


def _series_metrics_from_frame(
    metrics_df: pd.DataFrame,
    model_name: str | None,
) -> dict[str, dict[str, float]]:
    return series_metrics_from_frame(metrics_df, model_name)


def _aggregate_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]]
) -> dict[str, float]:
    return compute_aggregate_metrics(series_metrics_agg)


def _validate_temporal_ordering(df: pd.DataFrame) -> None:
    validate_temporal_ordering(df)


def _generate_cutoffs(
    all_dates: list[pd.Timestamp],
    n_windows: int,
    horizon: int,
    step: int,
    min_train_size: int,
    strategy: Literal["expanding", "sliding"],
) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    return generate_cutoffs(
        all_dates=all_dates,
        n_windows=n_windows,
        horizon=horizon,
        step=step,
        min_train_size=min_train_size,
        strategy=strategy,
    )


def cross_validation_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    horizon: int = 1,
    gap: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    return _cross_validation_split(
        df=df,
        n_splits=n_splits,
        horizon=horizon,
        gap=gap,
    )


def _reconcile_forecast(
    forecast_df: pd.DataFrame,
    hierarchy: Any,
    method: Any,
) -> pd.DataFrame:
    return reconcile_forecast(forecast_df, hierarchy, method)


def _build_window_covariates(
    dataset: TSDataset,
    task_spec: TaskSpec,
    cutoff_date: pd.Timestamp,
    panel_for_index: pd.DataFrame,
) -> Any:
    return build_window_covariates(dataset, task_spec, cutoff_date, panel_for_index)


def _compute_segment_metrics(
    series_metrics: dict[str, SeriesMetrics],
    dataset: TSDataset,
) -> dict[str, SegmentMetrics]:
    return compute_segment_metrics(series_metrics, dataset)


def _compute_temporal_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]],
    df: pd.DataFrame,
) -> dict[str, TemporalMetrics]:
    return compute_temporal_metrics(series_metrics_agg, df)
