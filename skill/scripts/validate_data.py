#!/usr/bin/env python3
"""
Validate time-series data format for tsagentkit.

Usage:
    python validate_data.py --input data.csv
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate data format for tsagentkit")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")

    args = parser.parse_args()

    try:
        from tsagentkit import validate
        from tsagentkit.core.errors import EContract
    except ImportError:
        print("Error: tsagentkit not installed. Run: pip install tsagentkit")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nHead:\n{df.head()}")

    # Validate
    print("\n" + "-" * 50)
    print("Validation:")
    print("-" * 50)

    try:
        validated = validate(df)
        print("Data format: VALID")
        print(f"Series count: {validated['unique_id'].nunique()}")
        print(f"Date range: {validated['ds'].min()} to {validated['ds'].max()}")
        return 0
    except EContract as e:
        print(f"Data format: INVALID")
        print(f"\nError: {e.message}")
        print(f"Fix: {e.fix_hint}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
