"""Tests for contracts/task_spec.py."""

import pytest
from pydantic import ValidationError

from tsagentkit.contracts import TaskSpec


class TestTaskSpecCreation:
    """Tests for TaskSpec creation."""

    def test_minimal_spec(self) -> None:
        """Test creating spec with minimal fields."""
        spec = TaskSpec(horizon=7, freq="D")
        assert spec.horizon == 7
        assert spec.freq == "D"
        assert spec.rolling_step == 7  # Defaults to horizon

    def test_spec_with_quantiles(self) -> None:
        """Test creating spec with quantiles."""
        spec = TaskSpec(
            horizon=14,
            freq="H",
            quantiles=[0.1, 0.5, 0.9],
        )
        assert spec.quantiles == [0.1, 0.5, 0.9]
        assert spec.season_length == 24  # Auto-inferred from "H"

    def test_spec_covariate_policy(self) -> None:
        """Test covariate policy options."""
        for policy in ["ignore", "known", "observed", "auto"]:
            spec = TaskSpec(horizon=7, freq="D", covariate_policy=policy)  # type: ignore
            assert spec.covariate_policy == policy


class TestTaskSpecValidation:
    """Tests for TaskSpec validation."""

    def test_horizon_must_be_positive(self) -> None:
        """Test horizon must be >= 1."""
        with pytest.raises(ValidationError):
            TaskSpec(horizon=0, freq="D")

        with pytest.raises(ValidationError):
            TaskSpec(horizon=-1, freq="D")

    def test_rolling_step_must_be_positive(self) -> None:
        """Test rolling_step must be >= 1 if provided."""
        with pytest.raises(ValidationError):
            TaskSpec(horizon=7, freq="D", rolling_step=0)

    def test_quantiles_must_be_between_0_and_1(self) -> None:
        """Test quantile validation."""
        with pytest.raises(ValidationError):
            TaskSpec(horizon=7, freq="D", quantiles=[0.0, 0.5])

        with pytest.raises(ValidationError):
            TaskSpec(horizon=7, freq="D", quantiles=[0.5, 1.0])

        with pytest.raises(ValidationError):
            TaskSpec(horizon=7, freq="D", quantiles=[-0.1])

    def test_quantiles_are_sorted_and_deduplicated(self) -> None:
        """Test quantiles are automatically sorted and deduplicated."""
        spec = TaskSpec(horizon=7, freq="D", quantiles=[0.9, 0.1, 0.5, 0.1])
        assert spec.quantiles == [0.1, 0.5, 0.9]

    def test_valid_date_order(self) -> None:
        """Test valid_from must be before valid_until."""
        with pytest.raises(ValidationError):
            TaskSpec(
                horizon=7,
                freq="D",
                valid_from="2024-01-15",
                valid_until="2024-01-01",
            )


class TestSeasonLengthInference:
    """Tests for season length inference."""

    def test_daily_frequency(self) -> None:
        """Test season length for daily frequency."""
        spec = TaskSpec(horizon=7, freq="D")
        assert spec.season_length == 7

    def test_hourly_frequency(self) -> None:
        """Test season length for hourly frequency."""
        spec = TaskSpec(horizon=24, freq="H")
        assert spec.season_length == 24

    def test_weekly_frequency(self) -> None:
        """Test season length for weekly frequency."""
        spec = TaskSpec(horizon=4, freq="W")
        assert spec.season_length == 52

    def test_monthly_frequency(self) -> None:
        """Test season length for monthly frequency."""
        spec = TaskSpec(horizon=3, freq="M")
        assert spec.season_length == 12

    def test_quarterly_frequency(self) -> None:
        """Test season length for quarterly frequency."""
        spec = TaskSpec(horizon=2, freq="Q")
        assert spec.season_length == 4

    def test_business_day_frequency(self) -> None:
        """Test season length for business day frequency."""
        spec = TaskSpec(horizon=5, freq="B")
        assert spec.season_length == 5

    def test_prefixed_frequency(self) -> None:
        """Test season length with numeric prefix."""
        spec = TaskSpec(horizon=7, freq="2D")
        assert spec.season_length == 7  # Still weekly seasonality

    def test_unknown_frequency(self) -> None:
        """Test unknown frequency returns None."""
        spec = TaskSpec(horizon=7, freq="X")
        assert spec.season_length is None


class TestTaskSpecImmutability:
    """Tests for TaskSpec immutability."""

    def test_frozen_model(self) -> None:
        """Test that spec is immutable after creation."""
        spec = TaskSpec(horizon=7, freq="D")
        with pytest.raises(ValidationError):
            spec.horizon = 14  # type: ignore

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            TaskSpec(horizon=7, freq="D", extra_field="not_allowed")  # type: ignore


class TestTaskSpecSignatures:
    """Tests for hash and signature generation."""

    def test_model_hash(self) -> None:
        """Test hash generation."""
        spec1 = TaskSpec(horizon=7, freq="D")
        spec2 = TaskSpec(horizon=7, freq="D")
        spec3 = TaskSpec(horizon=14, freq="D")

        assert spec1.model_hash() == spec2.model_hash()
        assert spec1.model_hash() != spec3.model_hash()

    def test_hash_is_hex(self) -> None:
        """Test hash is hexadecimal string."""
        spec = TaskSpec(horizon=7, freq="D")
        hash_val = spec.model_hash()
        assert len(hash_val) == 16
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_to_signature(self) -> None:
        """Test human-readable signature."""
        spec = TaskSpec(horizon=7, freq="D")
        sig = spec.to_signature()
        assert "H=7" in sig
        assert "f=D" in sig
        assert "TaskSpec(" in sig

    def test_signature_with_quantiles(self) -> None:
        """Test signature includes quantiles."""
        spec = TaskSpec(horizon=7, freq="D", quantiles=[0.1, 0.9])
        sig = spec.to_signature()
        assert "q=[0.10,0.90]" in sig

    def test_signature_with_season(self) -> None:
        """Test signature includes season length."""
        spec = TaskSpec(horizon=7, freq="D")
        sig = spec.to_signature()
        assert "s=7" in sig
