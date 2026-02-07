"""CLI entry point for tsagentkit.

Enables ``python -m tsagentkit <command>`` usage.

Subcommands:
    doctor   — Environment check: deps, adapters, system readiness.
    describe — Machine-readable API schema (JSON to stdout).
    version  — Print tsagentkit version.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys


def _check_import(module_name: str) -> tuple[bool, str | None]:
    """Try importing a module and return (success, version_or_none)."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", None))
        return True, str(version) if version is not None else "installed"
    except ImportError:
        return False, None


def _cmd_doctor() -> int:
    """Run environment diagnostics."""
    import tsagentkit

    print(f"tsagentkit {tsagentkit.__version__}")
    print(f"Python {sys.version}")
    print()

    # Core dependencies
    core_deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
        ("scipy", "scipy"),
        ("statsforecast", "statsforecast"),
        ("utilsforecast", "utilsforecast"),
    ]

    print("Core dependencies:")
    all_core_ok = True
    for display_name, module_name in core_deps:
        ok, version = _check_import(module_name)
        status = f"  {version}" if ok else "  NOT INSTALLED"
        marker = "ok" if ok else "MISSING"
        print(f"  [{marker:>7s}] {display_name}{status}")
        if not ok:
            all_core_ok = False

    print()

    # Optional dependencies — TSFM tier
    tsfm_deps = [
        ("torch", "torch"),
        ("chronos-forecasting", "chronos"),
        ("uni2ts (moirai)", "uni2ts"),
        ("timesfm", "timesfm"),
        ("gluonts", "gluonts"),
    ]

    print("TSFM tier (pip install tsagentkit[tsfm]):")
    for display_name, module_name in tsfm_deps:
        ok, version = _check_import(module_name)
        status = f"  {version}" if ok else "  not installed"
        marker = "ok" if ok else "---"
        print(f"  [{marker:>7s}] {display_name}{status}")

    print()

    # Hierarchy tier
    print("Hierarchy tier (pip install tsagentkit[hierarchy]):")
    ok, version = _check_import("hierarchicalforecast")
    status = f"  {version}" if ok else "  not installed"
    marker = "ok" if ok else "---"
    print(f"  [{marker:>7s}] hierarchicalforecast{status}")

    print()

    # Features tier
    features_deps = [
        ("tsfeatures", "tsfeatures"),
        ("tsfresh", "tsfresh"),
        ("sktime", "sktime"),
    ]

    print("Features tier (pip install tsagentkit[features]):")
    for display_name, module_name in features_deps:
        ok, version = _check_import(module_name)
        status = f"  {version}" if ok else "  not installed"
        marker = "ok" if ok else "---"
        print(f"  [{marker:>7s}] {display_name}{status}")

    print()

    # TSFM adapter registration
    print("TSFM adapter status:")
    try:
        from tsagentkit.models.adapters import AdapterRegistry

        registered = AdapterRegistry.list_available()
        if not registered:
            print("  No adapters registered.")
        else:
            for name in registered:
                is_available, reason = AdapterRegistry.check_availability(name)
                if is_available:
                    print(f"  [{' ok':>7s}] {name}  available")
                else:
                    print(f"  [{'---':>7s}] {name}  {reason}")
    except Exception as exc:
        print(f"  Could not query adapter registry: {exc}")

    print()

    # Verdict
    if all_core_ok:
        print("All systems go.")
    else:
        print("WARNING: Some core dependencies are missing. Install with:")
        print("  pip install tsagentkit")

    return 0


def _cmd_describe() -> int:
    """Print machine-readable API schema as JSON."""
    from tsagentkit.discovery import describe

    info = describe()
    json.dump(info, sys.stdout, indent=2, default=str)
    print()  # trailing newline
    return 0


def _cmd_version() -> int:
    """Print version string."""
    import tsagentkit

    print(tsagentkit.__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tsagentkit",
        description="tsagentkit — Robust execution engine for time-series forecasting agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="Environment check: deps, adapters, readiness")
    subparsers.add_parser("describe", help="Machine-readable API schema (JSON)")
    subparsers.add_parser("version", help="Print version")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _cmd_doctor()
    elif args.command == "describe":
        return _cmd_describe()
    elif args.command == "version":
        return _cmd_version()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
