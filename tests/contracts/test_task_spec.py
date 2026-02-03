"""Tests for contracts/task_spec.py (PRD-aligned)."""

import pytest
from pydantic import ValidationError

from tsagentkit.contracts import BacktestSpec, ForecastContract, TaskSpec


class TestTaskSpecCreation:
    """Tests for TaskSpec creation."""

    def test_minimal_spec(self) -> None:
        """Test creating spec with minimal fields."""
        spec = TaskSpec(h=7, freq="D")
        assert spec.h == 7
        assert spec.horizon == 7
        assert spec.freq == "D"
        assert spec.backtest.h == 7

    def test_spec_with_quantiles(self) -> None:
        """Test creating spec with forecast contract."""
        spec = TaskSpec(
            h=14,
            freq="H",
            forecast_contract=ForecastContract(quantiles=[0.1, 0.5, 0.9]),
        )
        assert spec.quantiles == [0.1, 0.5, 0.9]
        assert spec.season_length == 24

    def test_spec_covariate_policy(self) -> None:
        """Test covariate policy options."""
        for policy in ["ignore", "known", "observed", "auto", "spec"]:
            spec = TaskSpec(h=7, freq="D", covariate_policy=policy)  # type: ignore
            assert spec.covariate_policy == policy


class TestTaskSpecValidation:
    """Tests for TaskSpec validation."""

    def test_h_must_be_positive(self) -> None:
        """Test h must be >= 1."""
        with pytest.raises(ValidationError):
            TaskSpec(h=0, freq="D")

        with pytest.raises(ValidationError):
            TaskSpec(h=-1, freq="D")

    def test_backtest_step_positive(self) -> None:
        """Test backtest step must be >= 1 if provided."""
        with pytest.raises(ValidationError):
            TaskSpec(h=7, freq="D", backtest=BacktestSpec(h=1, step=0))


class TestSeasonLengthInference:
    """Tests for season length inference (compatibility)."""

    def test_daily_frequency(self) -> None:
        spec = TaskSpec(h=7, freq="D")
        assert spec.season_length == 7

    def test_hourly_frequency(self) -> None:
        spec = TaskSpec(h=24, freq="H")
        assert spec.season_length == 24


class TestTaskSpecImmutability:
    """Tests for TaskSpec immutability."""

    def test_frozen_model(self) -> None:
        spec = TaskSpec(h=7, freq="D")
        with pytest.raises(ValidationError):
            spec.h = 14  # type: ignore

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TaskSpec(h=7, freq="D", extra_field="not_allowed")  # type: ignore


class TestTaskSpecHashing:
    """Tests for hash generation."""

    def test_model_hash(self) -> None:
        spec1 = TaskSpec(h=7, freq="D")
        spec2 = TaskSpec(h=7, freq="D")
        spec3 = TaskSpec(h=14, freq="D")

        assert spec1.model_hash() == spec2.model_hash()
        assert spec1.model_hash() != spec3.model_hash()
