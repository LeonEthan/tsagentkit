#!/usr/bin/env python3
"""
Quick forecast script for tsagentkit.

Usage:
    python forecast.py --input data.csv --h 7 --freq D --output forecast.csv
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run tsagentkit forecast")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--h", type=int, required=True, help="Forecast horizon")
    parser.add_argument("--freq", "-f", default="D", help="Frequency (D, H, M, etc.)")
    parser.add_argument("--output", "-o", default="forecast.csv", help="Output CSV file path")
    parser.add_argument("--quantiles", "-q", default="0.1,0.5,0.9", help="Quantiles (comma-separated)")

    args = parser.parse_args()

    try:
        from tsagentkit import forecast, ForecastConfig
    except ImportError:
        print("Error: tsagentkit not installed. Run: pip install tsagentkit")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Parse quantiles
    quantiles = tuple(float(q) for q in args.quantiles.split(","))

    # Create config
    config = ForecastConfig(
        h=args.h,
        freq=args.freq,
        quantiles=quantiles,
    )

    # Run forecast
    print(f"Running forecast (h={args.h}, freq={args.freq})...")
    result = forecast(df, config=config)

    # Save output
    result.df.to_csv(args.output, index=False)
    print(f"Forecast saved to {args.output}")
    print(f"\nForecast head:\n{result.df.head()}")


if __name__ == "__main__":
    main()
