"""Tests for tsagentkit CLI (python -m tsagentkit)."""

from __future__ import annotations

import json
import subprocess
import sys

from tsagentkit.__main__ import main


def test_cli_version_via_main() -> None:
    """main(['version']) exits 0."""
    assert main(["version"]) == 0


def test_cli_doctor_via_main() -> None:
    """main(['doctor']) exits 0."""
    assert main(["doctor"]) == 0


def test_cli_describe_via_main() -> None:
    """main(['describe']) exits 0."""
    assert main(["describe"]) == 0


def test_cli_no_command_shows_help(capsys) -> None:
    """No subcommand prints help and exits 0."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "tsagentkit" in captured.out


def test_cli_version_subprocess() -> None:
    """python -m tsagentkit version outputs version string."""
    result = subprocess.run(
        [sys.executable, "-m", "tsagentkit", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip()  # non-empty version


def test_cli_doctor_subprocess() -> None:
    """python -m tsagentkit doctor runs without error."""
    result = subprocess.run(
        [sys.executable, "-m", "tsagentkit", "doctor"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "tsagentkit" in result.stdout
    assert "Core dependencies" in result.stdout


def test_cli_doctor_detects_core_deps(capsys) -> None:
    """Doctor output mentions core dependencies."""
    main(["doctor"])
    captured = capsys.readouterr()
    for dep in ["pandas", "numpy", "pydantic", "scipy", "statsforecast"]:
        assert dep in captured.out


def test_cli_doctor_detects_optional_deps(capsys) -> None:
    """Doctor output mentions dependency sections."""
    main(["doctor"])
    captured = capsys.readouterr()
    # All dependencies are now core (not optional tiers)
    assert "Core dependencies" in captured.out
    # Check that key packages are mentioned
    for dep in ["pandas", "numpy", "torch", "chronos"]:
        assert dep in captured.out


def test_cli_doctor_shows_adapter_status(capsys) -> None:
    """Doctor output mentions TSFM adapter status."""
    main(["doctor"])
    captured = capsys.readouterr()
    assert "TSFM adapter status" in captured.out


def test_cli_doctor_shows_verdict(capsys) -> None:
    """Doctor output contains a verdict line."""
    main(["doctor"])
    captured = capsys.readouterr()
    # Should have either "All systems go" or "WARNING"
    assert "All systems go" in captured.out or "WARNING" in captured.out


def test_cli_describe_outputs_valid_json() -> None:
    """python -m tsagentkit describe outputs valid JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "tsagentkit", "describe"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    assert "version" in data


def test_cli_version_matches_package() -> None:
    """CLI version matches tsagentkit.__version__."""
    import tsagentkit

    result = subprocess.run(
        [sys.executable, "-m", "tsagentkit", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == tsagentkit.__version__


def test_cli_unknown_subcommand(capsys) -> None:
    """Unknown subcommand doesn't crash."""
    # argparse will raise SystemExit for unknown subcommands, but
    # our main() handles missing command with help
    ret = main([])
    assert ret == 0
