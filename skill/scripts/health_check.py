#!/usr/bin/env python3
"""
Health check script for tsagentkit.

Usage:
    python health_check.py
"""

import sys


def main():
    try:
        from tsagentkit import check_health, list_models
    except ImportError:
        print("Error: tsagentkit not installed. Run: pip install tsagentkit")
        sys.exit(1)

    print("=" * 50)
    print("tsagentkit Health Check")
    print("=" * 50)

    # Health report
    health = check_health()
    print(f"\n{health}")

    # List models
    print("\n" + "-" * 50)
    print("Available Models:")
    print("-" * 50)

    tsfm_models = list_models(tsfm_only=True)
    print(f"\nTSFM Models: {tsfm_models}")

    all_models = list_models(tsfm_only=False)
    print(f"All Models: {all_models}")

    # Exit code based on health
    if health.tsfm_available:
        print("\nStatus: OK - Ready for forecasting")
        return 0
    else:
        print("\nStatus: WARNING - No TSFMs available")
        print("Install TSFM dependencies for best results")
        return 1


if __name__ == "__main__":
    sys.exit(main())
