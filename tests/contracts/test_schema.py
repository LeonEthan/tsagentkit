"""Tests for contracts/schema.py."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.contracts import (
    EContractDuplicateKey,
    EContractInvalidType,
    EContractMissingColumn,
    EContractUnsorted,
    validate_contract,
)


class TestValidateContractValid:
    """Tests for valid data validation."""

    def test_valid_dataframe(self) -> None:
        """Test validation of a valid DataFrame."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        report = validate_contract(df)
        assert report.valid is True
        assert len(report.errors) == 0

    def test_valid_with_extra_columns(self) -> None:
        """Test validation with extra columns."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
            "y": [1.0, 2.0],
            "extra_col": ["x", "y"],
        })
        report = validate_contract(df)
        assert report.valid is True

    def test_valid_list_of_dicts(self) -> None:
        """Test validation with list of dicts."""
        data = [
            {"unique_id": "A", "ds": "2024-01-01", "y": 1.0},
            {"unique_id": "A", "ds": "2024-01-02", "y": 2.0},
        ]
        report = validate_contract(data)
        assert report.valid is True

    def test_valid_single_series(self) -> None:
        """Test validation with single series."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": np.random.randn(10),
        })
        report = validate_contract(df)
        assert report.valid is True


class TestValidateContractMissingColumns:
    """Tests for missing column errors."""

    def test_missing_unique_id(self) -> None:
        """Test error when unique_id is missing."""
        df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        })
        report = validate_contract(df)
        assert report.valid is False
        assert len(report.errors) == 1
        assert report.errors[0]["code"] == EContractMissingColumn.error_code
        assert "unique_id" in str(report.errors[0]["context"]["missing"])

    def test_missing_ds(self) -> None:
        """Test error when ds is missing."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "y": [1.0, 2.0, 3.0],
        })
        report = validate_contract(df)
        assert report.valid is False
        assert report.errors[0]["code"] == EContractMissingColumn.error_code
        assert "ds" in str(report.errors[0]["context"]["missing"])

    def test_missing_y(self) -> None:
        """Test error when y is missing."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
        })
        report = validate_contract(df)
        assert report.valid is False
        assert report.errors[0]["code"] == EContractMissingColumn.error_code
        assert "y" in str(report.errors[0]["context"]["missing"])

    def test_missing_all_columns(self) -> None:
        """Test error when all required columns are missing."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = validate_contract(df)
        assert report.valid is False
        assert len(report.errors[0]["context"]["missing"]) == 3


class TestValidateContractTypes:
    """Tests for column type validation."""

    def test_unique_id_converted_to_string(self) -> None:
        """Test unique_id is auto-converted to string."""
        df = pd.DataFrame({
            "unique_id": [1, 1, 2, 2],  # Integers
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        report = validate_contract(df)
        # Should be valid after conversion
        assert report.valid is True

    def test_ds_converted_to_datetime(self) -> None:
        """Test ds is auto-converted to datetime."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": ["2024-01-01", "2024-01-02"],  # Strings
            "y": [1.0, 2.0],
        })
        report = validate_contract(df)
        assert report.valid is True

    def test_y_must_be_numeric(self) -> None:
        """Test error when y is not numeric."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
            "y": ["a", "b"],  # Strings
        })
        report = validate_contract(df)
        assert report.valid is False
        assert report.errors[0]["code"] == EContractInvalidType.error_code

    def test_invalid_ds_format(self) -> None:
        """Test error when ds cannot be parsed as datetime."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": ["invalid", "dates"],
            "y": [1.0, 2.0],
        })
        report = validate_contract(df)
        assert report.valid is False
        assert report.errors[0]["code"] == EContractInvalidType.error_code


class TestValidateContractDuplicates:
    """Tests for duplicate detection."""

    def test_duplicate_keys(self) -> None:
        """Test error when duplicate (unique_id, ds) pairs exist."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A"],  # Duplicate A at 2024-01-01
            "ds": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "y": [1.0, 2.0, 3.0],
        })
        report = validate_contract(df)
        assert report.valid is False
        assert report.errors[0]["code"] == EContractDuplicateKey.error_code

    def test_multiple_duplicates(self) -> None:
        """Test error with multiple duplicate pairs."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        report = validate_contract(df)
        assert report.valid is False
        # num_duplicates counts total rows that are duplicates (4)
        assert "4" in str(report.errors[0]["context"]["num_duplicates"])

    def test_no_duplicates_different_series(self) -> None:
        """Test no error when same date in different series."""
        df = pd.DataFrame({
            "unique_id": ["A", "B"],
            "ds": ["2024-01-01", "2024-01-01"],  # Same date, different series
            "y": [1.0, 2.0],
        })
        report = validate_contract(df)
        assert report.valid is True


class TestValidateContractSorting:
    """Tests for sorting validation."""

    def test_unsorted_data(self) -> None:
        """Test error when data is not sorted."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": ["2024-01-02", "2024-01-01"],  # Not sorted
            "y": [1.0, 2.0],
        })
        report = validate_contract(df)
        assert report.valid is False
        assert report.errors[0]["code"] == EContractUnsorted.error_code

    def test_unsorted_across_series(self) -> None:
        """Test error when series are not grouped."""
        df = pd.DataFrame({
            "unique_id": ["A", "B", "A"],
            "ds": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "y": [1.0, 2.0, 3.0],
        })
        report = validate_contract(df)
        assert report.valid is False

    def test_properly_sorted(self) -> None:
        """Test no error when properly sorted."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        report = validate_contract(df)
        assert report.valid is True


class TestValidateContractStats:
    """Tests for validation statistics."""

    def test_basic_stats(self) -> None:
        """Test that stats are computed."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        report = validate_contract(df)
        assert report.stats["num_rows"] == 4
        assert report.stats["num_series"] == 2
        assert "y_stats" in report.stats
        assert "date_range" in report.stats
        assert "series_lengths" in report.stats

    def test_series_length_stats(self) -> None:
        """Test series length statistics."""
        df = pd.DataFrame({
            "unique_id": ["A", "A", "A", "B", "B"],
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        report = validate_contract(df)
        lengths = report.stats["series_lengths"]
        assert lengths["min"] == 2
        assert lengths["max"] == 3
        assert lengths["mean"] == 2.5

    def test_y_stats(self) -> None:
        """Test y column statistics."""
        df = pd.DataFrame({
            "unique_id": ["A"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        report = validate_contract(df)
        y_stats = report.stats["y_stats"]
        assert y_stats["min"] == 1.0
        assert y_stats["max"] == 5.0
        assert y_stats["mean"] == 3.0


class TestValidateContractRaiseIfErrors:
    """Tests for raise_if_errors method."""

    def test_no_error_on_valid(self) -> None:
        """Test no exception when valid."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
            "y": [1.0, 2.0],
        })
        report = validate_contract(df)
        report.raise_if_errors()  # Should not raise

    def test_raises_on_invalid(self) -> None:
        """Test exception is raised when errors exist."""
        df = pd.DataFrame({
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
            # Missing y column
        })
        report = validate_contract(df)
        with pytest.raises(EContractMissingColumn):
            report.raise_if_errors()


class TestValidateContractEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self) -> None:
        """Test validation with empty DataFrame."""
        df = pd.DataFrame(columns=["unique_id", "ds", "y"])
        report = validate_contract(df)
        # Should fail due to missing columns check
        assert report.valid is True  # All columns present

    def test_single_row(self) -> None:
        """Test validation with single row."""
        df = pd.DataFrame({
            "unique_id": ["A"],
            "ds": ["2024-01-01"],
            "y": [1.0],
        })
        report = validate_contract(df)
        assert report.valid is True

    def test_invalid_input_type(self) -> None:
        """Test validation with non-convertible input."""
        report = validate_contract("invalid string")
        assert report.valid is False

    def test_dict_input(self) -> None:
        """Test validation with dict input."""
        data = {
            "unique_id": ["A", "A"],
            "ds": ["2024-01-01", "2024-01-02"],
            "y": [1.0, 2.0],
        }
        report = validate_contract(data)
        assert report.valid is True
