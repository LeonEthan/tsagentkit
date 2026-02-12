"""Phase 6 documentation consistency checks."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_skill_docs_are_mirrored_in_package_tree() -> None:
    root = _repo_root()
    mirrored_pairs = [
        ("skill/README.md", "src/tsagentkit/skill/README.md"),
        ("skill/recipes.md", "src/tsagentkit/skill/recipes.md"),
        ("skill/tool_map.md", "src/tsagentkit/skill/tool_map.md"),
        ("skill/QUICKSTART.md", "src/tsagentkit/skill/QUICKSTART.md"),
        ("skill/TROUBLESHOOTING.md", "src/tsagentkit/skill/TROUBLESHOOTING.md"),
    ]

    for left, right in mirrored_pairs:
        left_text = (root / left).read_text(encoding="utf-8")
        right_text = (root / right).read_text(encoding="utf-8")
        assert left_text == right_text, f"Skill docs diverged: {left} != {right}"


def test_phase4_phase5_apis_are_documented() -> None:
    root = _repo_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    skill_readme = (root / "skill" / "README.md").read_text(encoding="utf-8")
    tool_map = (root / "skill" / "tool_map.md").read_text(encoding="utf-8")

    # Core lifecycle APIs that should be documented in main README
    for symbol in [
        "list_adapter_capabilities",
        "save_run_artifact",
        "load_run_artifact",
        "validate_run_artifact_for_serving",
        "replay_forecast_from_artifact",
    ]:
        assert symbol in readme

    for symbol in [
        "save_run_artifact",
        "load_run_artifact",
        "validate_run_artifact_for_serving",
        "replay_forecast_from_artifact",
    ]:
        assert symbol in skill_readme
        assert symbol in tool_map
