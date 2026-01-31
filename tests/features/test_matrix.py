"""Tests for FeatureMatrix dataclass."""

import numpy as np
import pandas as pd
import pytest

from tsagentkit.features.matrix import FeatureMatrix


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "unique_id": ["A", "A", "B", "B"],
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
        "y": [1.0, 2.0, 3.0, 4.0],
        "feature_1": [0.1, 0.2, 0.3, 0.4],
        "feature_2": [1, 2, 3, 4],
    })


class TestFeatureMatrixCreation:
    """Test FeatureMatrix creation and validation."""

    def test_valid_creation(self, sample_data):
        """Test creating a valid FeatureMatrix."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1", "feature_2"],
        )
        assert matrix.config_hash == "abc123"
        assert matrix.feature_cols == ["feature_1", "feature_2"]
        assert matrix.target_col == "y"

    def test_missing_required_column(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({
            "unique_id": ["A", "B"],
            "y": [1.0, 2.0],
            # Missing "ds"
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            FeatureMatrix(data=df, config_hash="abc123")

    def test_invalid_feature_column(self, sample_data):
        """Test that invalid feature columns raise ValueError."""
        with pytest.raises(ValueError, match="Feature columns not in data"):
            FeatureMatrix(
                data=sample_data,
                config_hash="abc123",
                feature_cols=["nonexistent"],
            )

    def test_invalid_known_covariate(self, sample_data):
        """Test that invalid known covariates raise ValueError."""
        with pytest.raises(ValueError, match="Known covariates not in data"):
            FeatureMatrix(
                data=sample_data,
                config_hash="abc123",
                known_covariates=["nonexistent"],
            )

    def test_invalid_observed_covariate(self, sample_data):
        """Test that invalid observed covariates raise ValueError."""
        with pytest.raises(ValueError, match="Observed covariates not in data"):
            FeatureMatrix(
                data=sample_data,
                config_hash="abc123",
                observed_covariates=["nonexistent"],
            )


class TestFeatureMatrixProperties:
    """Test FeatureMatrix properties and methods."""

    def test_signature(self, sample_data):
        """Test signature property."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123def456",
            feature_cols=["feature_1", "feature_2"],
        )
        assert matrix.signature == "FeatureMatrix(c=abc123def456,n=2)"

    def test_to_pandas(self, sample_data):
        """Test to_pandas method returns a copy."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1", "feature_2"],
        )
        df = matrix.to_pandas()
        assert df is not matrix.data
        pd.testing.assert_frame_equal(df, matrix.data)

    def test_get_feature_data(self, sample_data):
        """Test get_feature_data method."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1", "feature_2"],
        )
        features = matrix.get_feature_data()
        assert list(features.columns) == ["feature_1", "feature_2"]
        assert len(features) == 4

    def test_get_target_data(self, sample_data):
        """Test get_target_data method."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1"],
        )
        target = matrix.get_target_data()
        assert list(target.values) == [1.0, 2.0, 3.0, 4.0]

    def test_get_covariate_data_known(self, sample_data):
        """Test get_covariate_data with known covariates."""
        sample_data["holiday"] = [0, 1, 0, 1]
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1"],
            known_covariates=["holiday"],
        )
        covs = matrix.get_covariate_data("known")
        assert list(covs.columns) == ["holiday"]

    def test_get_covariate_data_empty(self, sample_data):
        """Test get_covariate_data when no covariates of type exist."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1"],
        )
        covs = matrix.get_covariate_data("known")
        assert covs.empty


class TestFeatureMatrixValidation:
    """Test FeatureMatrix validation method."""

    def test_valid_data_no_issues(self, sample_data):
        """Test validation with clean data returns empty list."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1", "feature_2"],
        )
        issues = matrix.validate()
        assert issues == []

    def test_nulls_in_features(self, sample_data):
        """Test validation detects nulls in features."""
        sample_data.loc[0, "feature_1"] = np.nan
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1"],
        )
        issues = matrix.validate()
        assert len(issues) == 1
        assert "nulls" in issues[0]

    def test_infinite_values(self, sample_data):
        """Test validation detects infinite values."""
        sample_data.loc[0, "feature_1"] = np.inf
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1"],
        )
        issues = matrix.validate()
        assert len(issues) == 1
        assert "infinite" in issues[0]


class TestFeatureMatrixImmutability:
    """Test that FeatureMatrix is effectively immutable."""

    def test_data_is_copied_in_methods(self, sample_data):
        """Test that methods return copies, not references."""
        matrix = FeatureMatrix(
            data=sample_data,
            config_hash="abc123",
            feature_cols=["feature_1"],
        )
        # to_pandas should return a copy
        df1 = matrix.to_pandas()
        df2 = matrix.to_pandas()
        assert df1 is not df2

        # get_feature_data should return a copy
        f1 = matrix.get_feature_data()
        f2 = matrix.get_feature_data()
        assert f1 is not f2
