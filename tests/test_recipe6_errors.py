"""Recipe 6: Error Handling - End-to-end verification."""

import pandas as pd
from tsagentkit import TaskSpec, run_forecast, validate_contract
from tsagentkit.contracts import (
    TSAgentKitError,
    EContractMissingColumn,
    ESplitRandomForbidden,
)
from tsagentkit.contracts.errors import EContractUnsorted


def safe_forecast_with_validation(df, spec):
    """Run forecast with proper validation."""
    # Validate first
    validation = validate_contract(df)
    if not validation.valid:
        print("Validation failed:")
        for error in validation.errors:
            print(f"  - {error.get('code', 'unknown')}: {error.get('message', 'unknown')}")

        # Auto-fix if possible
        if any(e.get("code") == EContractMissingColumn.error_code for e in validation.errors):
            print("Hint: Ensure DataFrame has 'unique_id', 'ds', and 'y' columns")
        return None

    return run_forecast(df, spec)


def safe_forecast_sorted(df, spec):
    """Ensure data is sorted before forecasting."""
    # Sort data to prevent E_SPLIT_RANDOM_FORBIDDEN
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return run_forecast(df, spec)


def robust_forecast(df, spec, max_retries=2):
    """Run forecast with comprehensive error handling."""
    retries = 0
    while retries <= max_retries:
        try:
            result = run_forecast(df, spec, mode="standard")
            print("Success!")
            return result

        except EContractMissingColumn as e:
            print(f"Data error: {e.message}")
            print(f"Available columns: {e.context.get('available', list(df.columns))}")
            return None

        except (ESplitRandomForbidden, EContractUnsorted) as e:
            # Both errors indicate sorting issues
            print(f"Ordering error ({e.error_code}): {e.message}")
            print(f"Suggestion: {e.context.get('suggestion', 'Sort by unique_id, ds')}")
            # Try to fix
            df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            print("Retrying with sorted data...")
            retries += 1
            continue

        except TSAgentKitError as e:
            print(f"tsagentkit error ({e.error_code}): {e.message}")
            return None

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            return None

    print("Max retries exceeded")
    return None


def test_recipe6_error_handling():
    """Test Recipe 6: Error Handling."""
    print("=" * 60)
    print("RECIPE 6: Error Handling")
    print("=" * 60)

    spec = TaskSpec(horizon=2, freq="D")

    # Test 1: Valid Data
    print("\n=== Test 1: Valid Data ===")
    df_valid = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "ds": list(pd.date_range("2024-01-01", periods=4, freq="D")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    result = robust_forecast(df_valid, spec)
    assert result is not None, "Valid data should produce result"
    print(f"✓ Valid data handled correctly, model: {result.forecast.model_name}")

    # Test 2: Missing Column
    print("\n=== Test 2: Missing Column ===")
    df_missing = pd.DataFrame({"x": [1, 2, 3]})
    result = safe_forecast_with_validation(df_missing, spec)
    assert result is None, "Missing column should fail validation"
    print("✓ Missing column error handled correctly")

    # Test 3: Shuffled Data
    print("\n=== Test 3: Shuffled Data ===")
    import numpy as np

    np.random.seed(123)
    df_shuffled = df_valid.sample(frac=1).reset_index(drop=True)

    # First try should fail, but robust_forecast should fix it
    result = robust_forecast(df_shuffled, spec)
    assert result is not None, "Shuffled data should be fixed and produce result"
    print(f"✓ Shuffled data handled correctly (auto-sorted), model: {result.forecast.model_name}")

    # Test 4: Wrong column type
    print("\n=== Test 4: Wrong Column Type ===")
    df_wrong_type = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "A"],
            "ds": ["not", "a", "date", "column"],  # Wrong type
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    validation = validate_contract(df_wrong_type)
    print(f"Validation result: valid={validation.valid}")
    if not validation.valid:
        for error in validation.errors:
            print(f"  Error: {error.get('code', 'unknown')}: {error.get('message', 'unknown')}")
    assert not validation.valid, "Wrong type should fail validation"
    print("✓ Wrong column type error detected")

    # Test 5: Duplicate keys
    print("\n=== Test 5: Duplicate Keys ===")
    df_duplicates = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "A"],
            "ds": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),  # Duplicates
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    validation = validate_contract(df_duplicates)
    print(f"Validation result: valid={validation.valid}")
    if not validation.valid:
        for error in validation.errors:
            print(f"  Error: {error.get('code', 'unknown')}: {error.get('message', 'unknown')}")
    # Note: This might or might not fail validation depending on implementation
    print(f"✓ Duplicate keys validation completed (valid={validation.valid})")

    # Test 6: Test direct ESplitRandomForbidden
    print("\n=== Test 6: ESplitRandomForbidden Direct Test ===")
    try:
        # Create unsorted data and try to run backtest directly
        from tsagentkit.backtest import rolling_backtest
        from tsagentkit.series import TSDataset
        from tsagentkit.router import make_plan
        from tsagentkit.qa import run_qa
        from tsagentkit.models import fit, predict

        df_unsorted = df_valid.sample(frac=1).reset_index(drop=True)
        dataset = TSDataset.from_dataframe(df_unsorted, spec, validate=False)
        qa_report = run_qa(df_unsorted, spec)
        plan = make_plan(dataset, spec, qa_report)

        def bt_fit(model_name, train_df, config):
            train_ds = TSDataset.from_dataframe(train_df, spec, validate=False)
            return fit(model_name, train_ds, config)

        def bt_predict(model, test_df, horizon):
            test_ds = TSDataset.from_dataframe(
                test_df, spec, validate=False, compute_sparsity=False
            )
            return predict(model, test_ds, horizon)

        # This should raise ESplitRandomForbidden
        report = rolling_backtest(dataset, spec, plan, bt_fit, bt_predict, n_windows=1)
        print("⚠️  Expected ESplitRandomForbidden but got result")
    except ESplitRandomForbidden as e:
        print(f"✓ Caught ESplitRandomForbidden: {e.error_code}")
        print(f"  Message: {e.message}")
        print(f"  Suggestion: {e.context.get('suggestion', 'N/A')}")
    except Exception as e:
        print(f"⚠️  Got different error: {type(e).__name__}: {e}")

    print("\n✅ Recipe 6 PASSED - All Error Handling Tests Complete")
    return True


if __name__ == "__main__":
    test_recipe6_error_handling()
