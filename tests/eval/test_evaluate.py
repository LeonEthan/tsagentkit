"""Tests for tsagentkit.eval – evaluate_forecasts, MetricFrame, ScoreSummary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsagentkit.eval import MetricFrame, ScoreSummary, evaluate_forecasts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_forecast_df(
    n_series: int = 2,
    n_steps: int = 5,
    n_models: int = 1,
    include_quantiles: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a deterministic long-format forecast DataFrame."""
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    model_names = [f"model_{i}" for i in range(n_models)]
    dates = pd.date_range("2024-01-01", periods=n_steps, freq="D")

    for uid_idx in range(n_series):
        uid = f"series_{uid_idx}"
        for model in model_names:
            for step, ds in enumerate(dates):
                y = float(10 + uid_idx + step)
                yhat = y + rng.normal(0, 0.5)
                row: dict = {
                    "unique_id": uid,
                    "ds": ds,
                    "y": y,
                    "model": model,
                    "yhat": yhat,
                    "cutoff": dates[0],
                }
                if include_quantiles:
                    row["q0.1"] = yhat - 2.0
                    row["q0.9"] = yhat + 2.0
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluateForecasts:
    """Tests for evaluate_forecasts()."""

    def test_returns_metric_frame_and_score_summary(self) -> None:
        df = _make_forecast_df()
        mf, ss = evaluate_forecasts(df)
        assert isinstance(mf, MetricFrame)
        assert isinstance(ss, ScoreSummary)

    def test_empty_df_returns_empty(self) -> None:
        empty = pd.DataFrame()
        mf, ss = evaluate_forecasts(empty)
        assert isinstance(mf, MetricFrame)
        assert isinstance(ss, ScoreSummary)
        assert mf.df.empty
        assert ss.df.empty

    def test_single_model(self) -> None:
        df = _make_forecast_df(n_models=1)
        mf, ss = evaluate_forecasts(df)
        assert isinstance(mf, MetricFrame)
        assert isinstance(ss, ScoreSummary)
        # If utilsforecast is available, summary should have rows
        if not ss.df.empty:
            models_in_summary = ss.df["model"].unique()
            assert len(models_in_summary) == 1

    def test_multiple_models(self) -> None:
        df = _make_forecast_df(n_models=3)
        mf, ss = evaluate_forecasts(df)
        assert isinstance(mf, MetricFrame)
        assert isinstance(ss, ScoreSummary)
        if not ss.df.empty:
            models_in_summary = ss.df["model"].unique()
            assert len(models_in_summary) == 3

    def test_with_quantile_columns(self) -> None:
        df = _make_forecast_df(include_quantiles=True)
        mf, ss = evaluate_forecasts(df)
        assert isinstance(mf, MetricFrame)
        assert isinstance(ss, ScoreSummary)

    def test_no_model_column_adds_default(self) -> None:
        """When model column is missing, evaluate_forecasts injects 'model'."""
        df = _make_forecast_df(n_models=1)
        df = df.drop(columns=["model"])
        mf, ss = evaluate_forecasts(df)
        assert isinstance(mf, MetricFrame)
        assert isinstance(ss, ScoreSummary)


class TestDataclasses:
    """Tests for MetricFrame and ScoreSummary dataclass construction."""

    def test_metric_frame_frozen(self) -> None:
        mf = MetricFrame(df=pd.DataFrame({"a": [1]}))
        assert isinstance(mf.df, pd.DataFrame)
        with pytest.raises(AttributeError):
            mf.df = pd.DataFrame()  # type: ignore[misc]

    def test_score_summary_frozen(self) -> None:
        ss = ScoreSummary(df=pd.DataFrame({"b": [2]}))
        assert isinstance(ss.df, pd.DataFrame)
        with pytest.raises(AttributeError):
            ss.df = pd.DataFrame()  # type: ignore[misc]
