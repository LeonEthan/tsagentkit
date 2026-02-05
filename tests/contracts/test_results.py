"""Tests for contracts/results.py."""

import pandas as pd
import pytest

from tsagentkit.contracts import (
    ForecastResult,
    ModelArtifact,
    Provenance,
    ValidationReport,
)
from tsagentkit.contracts.errors import EContractMissingColumn


class TestProvenance:
    """Tests for Provenance dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a Provenance object."""
        prov = Provenance(
            run_id="test-run-123",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="abc123",
            task_signature="def456",
            plan_signature="ghi789",
            model_signature="jkl012",
        )
        assert prov.run_id == "test-run-123"
        assert len(prov.qa_repairs) == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        prov = Provenance(
            run_id="run-1",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="sig1",
            task_signature="sig2",
            plan_signature="sig3",
            model_signature="sig4",
            qa_repairs=[{"type": "interpolate"}],
            fallbacks_triggered=[{"from": "tsfm", "to": "naive"}],
        )
        d = prov.to_dict()
        assert d["run_id"] == "run-1"
        assert d["qa_repairs"] == [{"type": "interpolate"}]
        assert len(d["fallbacks_triggered"]) == 1

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "run_id": "run-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "data_signature": "sig1",
            "task_signature": "sig2",
            "plan_signature": "sig3",
            "model_signature": "sig4",
        }
        prov = Provenance.from_dict(data)
        assert prov.run_id == "run-1"


class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    @pytest.fixture
    def sample_provenance(self) -> Provenance:
        """Create a sample provenance."""
        return Provenance(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            data_signature="sig1",
            task_signature="sig2",
            plan_signature="sig3",
            model_signature="sig4",
        )

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample forecast DataFrame."""
        return pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "model": ["Test"] * 4,
            "yhat": [1.0, 2.0, 3.0, 4.0],
            "q0.1": [0.9, 1.9, 2.9, 3.9],
            "q0.9": [1.1, 2.1, 3.1, 4.1],
        })

    def test_valid_creation(
        self, sample_df: pd.DataFrame, sample_provenance: Provenance
    ) -> None:
        """Test creating a valid ForecastResult."""
        result = ForecastResult(
            df=sample_df,
            provenance=sample_provenance,
            model_name="SeasonalNaive",
            horizon=2,
        )
        assert result.model_name == "SeasonalNaive"
        assert result.horizon == 2

    def test_missing_required_column(self, sample_provenance: Provenance) -> None:
        """Test error when required column is missing."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": ["2024-01-01"],
            "model": ["Test"],
            # Missing yhat
        })
        with pytest.raises(ValueError, match="missing columns"):
            ForecastResult(
                df=df,
                provenance=sample_provenance,
                model_name="Test",
                horizon=1,
            )

    def test_invalid_ds_type(self, sample_provenance: Provenance) -> None:
        """Test error when ds is not datetime."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": ["not-a-date"],  # String instead of datetime
            "model": ["Test"],
            "yhat": [1.0],
        })
        with pytest.raises(ValueError, match="datetime"):
            ForecastResult(
                df=df,
                provenance=sample_provenance,
                model_name="Test",
                horizon=1,
            )

    def test_get_quantile_columns(
        self, sample_df: pd.DataFrame, sample_provenance: Provenance
    ) -> None:
        """Test getting quantile column names."""
        result = ForecastResult(
            df=sample_df,
            provenance=sample_provenance,
            model_name="Test",
            horizon=2,
        )
        quantiles = result.get_quantile_columns()
        assert "q0.1" in quantiles
        assert "q0.9" in quantiles
        assert "q" not in quantiles  # Just 'q' is not a valid quantile column

    def test_get_series(
        self, sample_df: pd.DataFrame, sample_provenance: Provenance
    ) -> None:
        """Test getting forecast for specific series."""
        result = ForecastResult(
            df=sample_df,
            provenance=sample_provenance,
            model_name="Test",
            horizon=2,
        )
        series_a = result.get_series("A")
        assert len(series_a) == 2
        assert all(series_a["unique_id"] == "A")


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_valid_report(self) -> None:
        """Test a valid report."""
        report = ValidationReport(valid=True)
        assert report.valid is True
        assert report.has_errors() is False
        assert report.has_warnings() is False

    def test_report_with_errors(self) -> None:
        """Test report with errors."""
        report = ValidationReport(
            valid=False,
            errors=[{"code": "E_TEST", "message": "Test error"}],
        )
        assert report.valid is False
        assert report.has_errors() is True

    def test_report_with_warnings(self) -> None:
        """Test report with warnings."""
        report = ValidationReport(
            valid=True,
            warnings=[{"code": "W_TEST", "message": "Test warning"}],
        )
        assert report.has_warnings() is True

    def test_raise_if_errors_no_errors(self) -> None:
        """Test raise_if_errors when no errors."""
        report = ValidationReport(valid=True)
        report.raise_if_errors()  # Should not raise

    def test_raise_if_errors_with_errors(self) -> None:
        """Test raise_if_errors raises exception."""
        report = ValidationReport(
            valid=False,
            errors=[{
                "code": EContractMissingColumn.error_code,
                "message": "Missing column",
                "context": {},
            }],
        )
        with pytest.raises(EContractMissingColumn):
            report.raise_if_errors()

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        report = ValidationReport(
            valid=True,
            stats={"n_rows": 100, "n_series": 5},
        )
        d = report.to_dict()
        assert d["valid"] is True
        assert d["stats"]["n_rows"] == 100
        assert d["stats"]["n_series"] == 5


class TestModelArtifact:
    """Tests for ModelArtifact dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a ModelArtifact."""
        artifact = ModelArtifact(
            model="dummy-model",
            model_name="TestModel",
            config={"param": "value"},
        )
        assert artifact.model_name == "TestModel"
        assert artifact.config == {"param": "value"}

    def test_signature_auto_computed(self) -> None:
        """Test that signature is auto-computed from config."""
        artifact = ModelArtifact(
            model="dummy",
            model_name="Test",
            config={"a": 1, "b": 2},
        )
        assert artifact.signature != ""
        assert len(artifact.signature) == 16  # SHA-256 truncated

    def test_signature_provided(self) -> None:
        """Test that provided signature is used."""
        artifact = ModelArtifact(
            model="dummy",
            model_name="Test",
            config={"a": 1},
            signature="custom-sig",
        )
        assert artifact.signature == "custom-sig"
