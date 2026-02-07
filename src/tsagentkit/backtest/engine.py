"""Rolling window backtest engine.

Implements expanding and sliding window backtesting with strict
temporal integrity (no random splits allowed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from tsagentkit.contracts import CVFrame, ESplitRandomForbidden
from tsagentkit.covariates import AlignedDataset, align_covariates
from tsagentkit.eval import evaluate_forecasts
from tsagentkit.utils import drop_future_rows, normalize_quantile_columns

from .report import (
    BacktestReport,
    SegmentMetrics,
    SeriesMetrics,
    TemporalMetrics,
    WindowResult,
)

if TYPE_CHECKING:
    from tsagentkit.contracts import TaskSpec
    from tsagentkit.hierarchy import HierarchyStructure, ReconciliationMethod
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
    _validate_temporal_ordering(df)

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
    cutoffs = _generate_cutoffs(
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
                window_covariates = _build_window_covariates(
                    dataset=dataset,
                    task_spec=spec,
                    cutoff_date=pd.Timestamp(cutoff_date),
                    panel_for_index=train_df,
                )
            except Exception as e:
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
                model = _call_with_optional_kwargs(
                    fit_func,
                    train_ds,
                    plan,
                    covariates=window_covariates,
                )
            except Exception as e:
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
                predictions = _call_with_optional_kwargs(
                    predict_func,
                    train_ds,
                    model,
                    spec,
                    covariates=window_covariates,
                )
            except Exception as e:
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
                predictions = _reconcile_forecast(
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

            window_metrics = _summary_to_metrics(summary.df, model_name)
            series_window_metrics = _series_metrics_from_frame(metric_frame.df, model_name)
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
    aggregate_metrics = _aggregate_metrics(series_metrics_agg)

    # Create series metrics
    series_metrics: dict[str, SeriesMetrics] = {}
    for uid, metrics_list in series_metrics_agg.items():
        metric_names: set[str] = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        avg_metrics = {
            k: np.mean([m[k] for m in metrics_list if not np.isnan(m.get(k, np.nan))])
            for k in metric_names
        }
        series_metrics[uid] = SeriesMetrics(
            series_id=uid,
            metrics=avg_metrics,
            num_windows=len(metrics_list),
        )

    # Compute segment metrics (by sparsity class) if sparsity profile available
    segment_metrics = _compute_segment_metrics(series_metrics, dataset)

    # Compute temporal metrics if datetime information available
    temporal_metrics = _compute_temporal_metrics(series_metrics_agg, df)

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
    if summary_df is None or summary_df.empty:
        return {}
    df = summary_df
    if "model" in df.columns:
        if model_name and model_name in df["model"].unique():
            df = df[df["model"] == model_name]
        else:
            df = df[df["model"] == df["model"].iloc[0]]
    if "metric" not in df.columns or "value" not in df.columns:
        return {}
    metrics = df.groupby("metric")["value"].mean()
    return {metric: float(value) for metric, value in metrics.items()}


def _series_metrics_from_frame(
    metrics_df: pd.DataFrame,
    model_name: str | None,
) -> dict[str, dict[str, float]]:
    if metrics_df is None or metrics_df.empty:
        return {}
    df = metrics_df
    if "model" in df.columns:
        if model_name and model_name in df["model"].unique():
            df = df[df["model"] == model_name]
        else:
            df = df[df["model"] == df["model"].iloc[0]]
    if "unique_id" not in df.columns or "metric" not in df.columns or "value" not in df.columns:
        return {}
    grouped = df.groupby(["unique_id", "metric"])["value"].mean().reset_index()
    result: dict[str, dict[str, float]] = {}
    for uid, group in grouped.groupby("unique_id"):
        result[uid] = {row["metric"]: float(row["value"]) for _, row in group.iterrows()}
    return result


def _aggregate_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]]
) -> dict[str, float]:
    """Aggregate metrics across all series and windows.

    Args:
        series_metrics_agg: Dict mapping series_id to list of metrics per window

    Returns:
        Dictionary of aggregated metrics
    """
    if not series_metrics_agg:
        return {}

    # Collect all metric names
    all_metric_names: set[str] = set()
    for metrics_list in series_metrics_agg.values():
        for metrics in metrics_list:
            all_metric_names.update(metrics.keys())

    # Aggregate each metric
    aggregated: dict[str, float] = {}
    for metric_name in all_metric_names:
        values = []
        for metrics_list in series_metrics_agg.values():
            for m in metrics_list:
                if metric_name in m and not np.isnan(m[metric_name]):
                    values.append(m[metric_name])

        if values:
            aggregated[metric_name] = float(np.mean(values))
        else:
            aggregated[metric_name] = float("nan")

    return aggregated


def _validate_temporal_ordering(df: pd.DataFrame) -> None:
    """Validate that data is temporally ordered (no shuffling).

    This is a critical guardrail to prevent data leakage.

    Args:
        df: DataFrame to validate

    Raises:
        ESplitRandomForbidden: If data appears to be randomly ordered
    """
    # Check if data is sorted by unique_id, ds
    expected_order = df.sort_values(["unique_id", "ds"]).index
    if not df.index.equals(expected_order):
        raise ESplitRandomForbidden(
            "Data must be sorted by (unique_id, ds). "
            "Random splits or shuffling is strictly forbidden.",
            context={
                "suggestion": "Ensure data is sorted: df.sort_values(['unique_id', 'ds'])",
            },
        )

    # Additional check: verify dates are monotonic within each series
    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]
        dates = pd.to_datetime(series["ds"])
        if not dates.is_monotonic_increasing:
            raise ESplitRandomForbidden(
                f"Dates for series {uid} are not monotonically increasing. "
                f"Data may be shuffled or contain time-travel.",
                context={"series_id": uid},
            )


def _generate_cutoffs(
    all_dates: list[pd.Timestamp],
    n_windows: int,
    horizon: int,
    step: int,
    min_train_size: int,
    strategy: Literal["expanding", "sliding"],
) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    """Generate cutoff dates for backtest windows.

    Args:
        all_dates: Sorted list of all dates in dataset
        n_windows: Number of windows
        horizon: Forecast horizon
        step: Step size between windows
        min_train_size: Minimum training set size
        strategy: "expanding" or "sliding"

    Returns:
        List of (cutoff_date, test_dates) tuples
    """
    cutoffs = []

    # Calculate starting point
    if strategy == "expanding":
        # Expanding window: each window adds more training data
        start_idx = min_train_size
        for i in range(n_windows):
            cutoff_idx = start_idx + i * step
            if cutoff_idx + horizon > len(all_dates):
                break

            cutoff_date = all_dates[cutoff_idx]
            test_dates = all_dates[cutoff_idx : cutoff_idx + horizon]
            cutoffs.append((cutoff_date, test_dates))

    elif strategy == "sliding":
        # Sliding window: fixed training size, slides forward
        total_needed = min_train_size + (n_windows - 1) * step + horizon
        if total_needed > len(all_dates):
            # Adjust n_windows
            n_windows = ((len(all_dates) - min_train_size - horizon) // step) + 1
            n_windows = max(n_windows, 0)

        for i in range(n_windows):
            train_end_idx = min_train_size + i * step
            cutoff_date = all_dates[train_end_idx]
            test_dates = all_dates[train_end_idx : train_end_idx + horizon]
            cutoffs.append((cutoff_date, test_dates))

    return cutoffs


def cross_validation_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    horizon: int = 1,
    gap: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate cross-validation splits with temporal validation.

    Random splits are strictly forbidden - this uses temporal splits only.

    Args:
        df: DataFrame with time series
        n_splits: Number of splits
        horizon: Forecast horizon
        gap: Gap between train and test

    Returns:
        List of (train_df, test_df) tuples

    Raises:
        ESplitRandomForbidden: If data is not temporally ordered
    """
    # Validate ordering
    _validate_temporal_ordering(df)

    splits = []
    dates = sorted(df["ds"].unique())

    # Calculate fold size
    fold_size = (len(dates) - horizon) // n_splits

    for i in range(n_splits):
        split_point = (i + 1) * fold_size

        train_end = dates[split_point - 1]
        test_start_idx = split_point + gap

        if test_start_idx + horizon > len(dates):
            break

        test_dates = dates[test_start_idx : test_start_idx + horizon]

        train_df = df[df["ds"] <= train_end].copy()
        test_df = df[df["ds"].isin(test_dates)].copy()

        splits.append((train_df, test_df))

    return splits


def _reconcile_forecast(
    forecast_df: pd.DataFrame,
    hierarchy: HierarchyStructure,
    method: str | ReconciliationMethod,
) -> pd.DataFrame:
    """Reconcile forecast to ensure hierarchy coherence.

    Args:
        forecast_df: Forecast DataFrame with columns [unique_id, ds, yhat]
        hierarchy: Hierarchy structure
        method: Reconciliation method name or enum

    Returns:
        Reconciled forecast DataFrame
    """
    from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

    # Convert method string to enum if needed
    if isinstance(method, str):
        method_map = {
            "bottom_up": ReconciliationMethod.BOTTOM_UP,
            "top_down": ReconciliationMethod.TOP_DOWN,
            "middle_out": ReconciliationMethod.MIDDLE_OUT,
            "ols": ReconciliationMethod.OLS,
            "wls": ReconciliationMethod.WLS,
            "min_trace": ReconciliationMethod.MIN_TRACE,
        }
        method = method_map.get(method, ReconciliationMethod.BOTTOM_UP)

    # Apply reconciliation
    reconciled = reconcile_forecasts(
        base_forecasts=forecast_df,
        structure=hierarchy,
        method=method,
    )

    return reconciled


def _build_window_covariates(
    dataset: TSDataset,
    task_spec: TaskSpec,
    cutoff_date: pd.Timestamp,
    panel_for_index: pd.DataFrame,
) -> AlignedDataset | None:
    """Align covariates for a specific backtest window."""
    if dataset.panel_with_covariates is None and dataset.covariate_bundle is None:
        return None

    ds_col = task_spec.panel_contract.ds_col
    y_col = task_spec.panel_contract.y_col

    if dataset.panel_with_covariates is not None:
        panel = dataset.panel_with_covariates.copy()
    else:
        panel = panel_for_index.copy()

    if y_col in panel.columns:
        panel.loc[panel[ds_col] >= cutoff_date, y_col] = np.nan

    return align_covariates(
        panel=panel,
        task_spec=task_spec,
        covariates=dataset.covariate_bundle,
    )


def _call_with_optional_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function with only supported keyword arguments."""
    if not kwargs:
        return func(*args)

    try:
        import inspect

        params = inspect.signature(func).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
        return func(*args, **accepted)
    except Exception:
        return func(*args, **kwargs)


def _compute_segment_metrics(
    series_metrics: dict[str, SeriesMetrics],
    dataset: TSDataset,
) -> dict[str, SegmentMetrics]:
    """Compute segment metrics grouped by sparsity class.

    Args:
        series_metrics: Dictionary of series_id to SeriesMetrics
        dataset: TSDataset with sparsity profile

    Returns:
        Dictionary of segment_name to SegmentMetrics
    """
    from collections import defaultdict

    segment_series: dict[str, list[str]] = defaultdict(list)
    segment_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)

    # Group series by sparsity class
    if dataset.sparsity_profile:
        for uid in series_metrics:
            classification = dataset.sparsity_profile.get_classification(uid)
            segment_name = classification.value
            segment_series[segment_name].append(uid)
            segment_metrics[segment_name].append(series_metrics[uid].metrics)
    else:
        # No sparsity profile, put all in "unknown" segment
        for uid, sm in series_metrics.items():
            segment_series["unknown"].append(uid)
            segment_metrics["unknown"].append(sm.metrics)

    # Aggregate metrics per segment
    result: dict[str, SegmentMetrics] = {}
    for segment_name, series_ids in segment_series.items():
        metrics_list = segment_metrics[segment_name]
        if not metrics_list:
            continue

        # Compute mean for each metric
        aggregated: dict[str, float] = {}
        metric_names: set[str] = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if not np.isnan(m.get(metric_name, np.nan))]
            if values:
                aggregated[metric_name] = float(np.mean(values))

        result[segment_name] = SegmentMetrics(
            segment_name=segment_name,
            series_ids=series_ids,
            metrics=aggregated,
            n_series=len(series_ids),
        )

    return result


def _select_primary_metric(
    series_metrics_agg: dict[str, list[dict[str, float]]],
    preferred: tuple[str, ...] = ("wape", "mae", "rmse", "smape", "mase"),
) -> str | None:
    metric_names: set[str] = set()
    for metrics_list in series_metrics_agg.values():
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
    for name in preferred:
        if name in metric_names:
            return name
    return next(iter(metric_names), None)


def _compute_temporal_metrics(
    series_metrics_agg: dict[str, list[dict[str, float]]],
    df: pd.DataFrame,
) -> dict[str, TemporalMetrics]:
    """Compute temporal metrics grouped by time dimensions.

    Args:
        series_metrics_agg: Dict mapping series_id to list of window metrics
        df: Original DataFrame with datetime information

    Returns:
        Dictionary of dimension to TemporalMetrics
    """
    result: dict[str, TemporalMetrics] = {}

    metric_name = _select_primary_metric(series_metrics_agg)
    if metric_name is None:
        return result

    # Parse dates
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Compute hour-of-day metrics
    df["hour"] = df["ds"].dt.hour
    hour_metrics: dict[str, dict[str, float]] = {}
    for hour in sorted(df["hour"].unique()):
        hour_str = str(hour)
        # Get series present in this hour
        hour_series = df[df["hour"] == hour]["unique_id"].unique()
        if len(hour_series) > 0:
            # Average metrics for series in this hour
            values = []
            for uid in hour_series:
                if uid in series_metrics_agg and series_metrics_agg[uid]:
                    avg_metric = np.mean([m.get(metric_name, np.nan) for m in series_metrics_agg[uid]])
                    if not np.isnan(avg_metric):
                        values.append(avg_metric)
            if values:
                hour_metrics[hour_str] = {metric_name: float(np.mean(values))}

    if hour_metrics:
        result["hour"] = TemporalMetrics(dimension="hour", metrics_by_value=hour_metrics)

    # Compute day-of-week metrics
    df["dayofweek"] = df["ds"].dt.dayofweek
    dow_metrics: dict[str, dict[str, float]] = {}
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for dow in sorted(df["dayofweek"].unique()):
        dow_str = dow_names[dow]
        # Get series present on this day of week
        dow_series = df[df["dayofweek"] == dow]["unique_id"].unique()
        if len(dow_series) > 0:
            # Average metrics for series on this day
            values = []
            for uid in dow_series:
                if uid in series_metrics_agg and series_metrics_agg[uid]:
                    avg_metric = np.mean([m.get(metric_name, np.nan) for m in series_metrics_agg[uid]])
                    if not np.isnan(avg_metric):
                        values.append(avg_metric)
            if values:
                dow_metrics[dow_str] = {metric_name: float(np.mean(values))}

    if dow_metrics:
        result["dayofweek"] = TemporalMetrics(dimension="dayofweek", metrics_by_value=dow_metrics)

    return result
