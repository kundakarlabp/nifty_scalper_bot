#!/usr/bin/env python3
"""File purpose: Build a focused validation plan for repository changes.
Key responsibilities: Classify changed files, select existing tests, and keep the full suite mandatory.
Operational constraints: Produce commands only; never execute broker or runtime code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

RULES = (
    ("streaming", ("/streaming/", "websocket"), ("tests/streaming", "tests/data")),
    (
        "market-data",
        ("/data/", "/instruments/", "instrument_manager.py"),
        ("tests/data", "tests/core", "tests/instruments"),
    ),
    (
        "execution",
        ("/execution/",),
        (
            "tests/execution",
            "tests/integration/test_canonical_bo_end_to_end.py",
            "tests/test_execution_path_contract.py",
        ),
    ),
    ("risk", ("/risk/",), ("tests/risk",)),
    ("strategy", ("/strategies/", "strategy_manager.py"), ("tests/strategies",)),
    ("notifications", ("/notifications/", "telegram"), ("tests/notifications",)),
    ("core", ("/core/",), ("tests/core", "tests/architecture")),
    ("dashboard", ("dashboard/",), ("tests/dashboard",)),
    (
        "deployment",
        ("deploy/", "ops/", "dockerfile", "railway.toml"),
        (
            "tests/core/test_release_guard.py",
            "tests/test_deployment_release_guard.py",
        ),
    ),
    (
        "agent-tooling",
        (
            "agents.md",
            ".agents/",
            "copilot-instructions.md",
            "repo_map.md",
            "scripts/agent_",
        ),
        (
            "tests/tools",
            "tests/architecture/test_agent_skills_catalog.py",
        ),
    ),
)


@dataclass(frozen=True)
class Plan:
    changed_files: tuple[str, ...]
    areas: tuple[str, ...]
    focused_tests: tuple[str, ...]
    commands: tuple[str, ...]
    full_suite_required: bool = True


def changed_from_git(root: Path, base_ref: str) -> list[str]:
    for command in (
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        ["git", "diff", "--name-only", "HEAD^", "HEAD"],
    ):
        try:
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if files:
            return files
    return []


def normalize_files(root: Path, files: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in files:
        if not item.strip():
            continue
        candidate = Path(item).expanduser()
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            relative = candidate
        normalized.append(relative.as_posix().lstrip("./"))
    return tuple(dict.fromkeys(normalized))


def build(root: Path, files: Sequence[str]) -> Plan:
    normalized = normalize_files(root, files)
    lowered = [item.lower() for item in normalized]
    areas: list[str] = []
    tests: list[str] = []
    for area, markers, candidates in RULES:
        if any(any(marker in path for marker in markers) for path in lowered):
            areas.append(area)
            tests.extend(
                candidate for candidate in candidates if (root / candidate).exists()
            )
    if any(path.startswith("src/") for path in normalized):
        tests.extend(
            candidate
            for candidate in (
                "tests/architecture/test_canonical_bo_ownership.py",
                "tests/test_execution_path_contract.py",
            )
            if (root / candidate).exists()
        )
    tests = list(dict.fromkeys(tests))
    commands = ["python -m compileall -q src dashboard"]
    commands.append(
        "python -m pytest -q " + " ".join(tests)
        if tests
        else "python -m pytest -q tests/architecture"
    )
    commands.append("python -m pytest -q")
    return Plan(
        normalized,
        tuple(areas or ["unclassified"]),
        tuple(tests),
        tuple(commands),
    )


def markdown(plan: Plan) -> str:
    lines = ["# Agent Validation Plan", "", "## Changed files", ""]
    lines.extend(f"- `{item}`" for item in plan.changed_files)
    if not plan.changed_files:
        lines.append("- No changed files detected.")
    lines.extend(
        [
            "",
            "## Areas",
            "",
            *(f"- `{item}`" for item in plan.areas),
            "",
            "## Focused tests",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in plan.focused_tests)
    if not plan.focused_tests:
        lines.append("- Architecture checks are the safe minimum.")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            *plan.commands,
            "```",
            "",
            "> Focused checks accelerate feedback; the complete suite and final-head CI remain mandatory before squash merge.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not (root / "src").exists() or not (root / "tests").exists():
        print(f"ERROR: {root} is not the repository root", file=sys.stderr)
        return 2
    plan = build(root, args.files or changed_from_git(root, args.base_ref))
    output = (
        json.dumps(asdict(plan), indent=2)
        if args.format == "json"
        else markdown(plan)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output.rstrip() + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
