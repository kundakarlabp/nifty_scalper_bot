#!/usr/bin/env python3
"""File purpose: Build or execute a focused validation plan for repository changes.
Key responsibilities: Classify changed files, select existing tests, and keep the full suite mandatory before merge.
Operational constraints: Never execute broker or runtime entry points; run only
repository quality, compile, and test commands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
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
            "agent_start_here.md",
            "agent_tooling_design.md",
            "ai_optimization_workflow.md",
            "chatgpt_code_workflow.md",
            "engineering_failure_patterns.md",
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
    base_ref: str = "origin/main"


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


def build(
    root: Path,
    files: Sequence[str],
    *,
    base_ref: str = "origin/main",
) -> Plan:
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
    commands = ["python -m compileall -q src dashboard scripts"]
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
        base_ref=base_ref,
    )


def commands_for_run(plan: Plan, scope: str) -> tuple[str, ...]:
    """Return the generated validation ring requested by the caller."""
    if scope == "focused":
        return plan.commands[:-1]
    if scope == "full":
        return plan.commands
    raise ValueError(f"Unsupported validation scope: {scope}")


def _changed_python_files(plan: Plan) -> tuple[str, ...]:
    """Return changed Python paths covered by the repository quality gate."""
    prefixes = ("src/", "dashboard/", "scripts/", "tests/")
    return tuple(
        path
        for path in plan.changed_files
        if path.endswith(".py") and path.startswith(prefixes)
    )


def run_quality_checks(root: Path, plan: Plan) -> int:
    """Run the repository's delta-aware quality checks before compile/tests."""
    python_files = _changed_python_files(plan)
    checkers = tuple(
        root / path
        for path in (
            "scripts/check_changed_ruff.py",
            "scripts/check_changed_black.py",
            "scripts/check_changed_mypy.py",
        )
        if (root / path).exists()
    )
    if not python_files or not checkers:
        return 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="agent-changed-python-",
        suffix=".txt",
        delete=False,
    ) as manifest:
        manifest.write("\n".join(python_files) + "\n")
        manifest_path = Path(manifest.name)

    try:
        for checker in checkers:
            argv = [
                sys.executable,
                str(checker),
                "--base",
                plan.base_ref,
                "--files-from",
                str(manifest_path),
            ]
            print(
                "+ " + " ".join(shlex.quote(part) for part in argv),
                file=sys.stderr,
            )
            try:
                result = subprocess.run(
                    argv,
                    cwd=root,
                    check=False,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                )
            except OSError as exc:
                print(
                    f"ERROR: failed to execute {checker.name!r}: {exc}",
                    file=sys.stderr,
                )
                return 2
            if result.returncode != 0:
                print(
                    "ERROR: changed-file quality check failed "
                    f"with exit code {result.returncode}: {checker.name}",
                    file=sys.stderr,
                )
                return result.returncode
    finally:
        manifest_path.unlink(missing_ok=True)

    return 0


def run_plan(root: Path, plan: Plan, scope: str) -> int:
    """Run changed-file quality checks, then the requested compile/test ring."""
    quality_result = run_quality_checks(root, plan)
    if quality_result != 0:
        return quality_result

    for command in commands_for_run(plan, scope):
        argv = shlex.split(command)
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        print(f"+ {command}", file=sys.stderr)
        try:
            result = subprocess.run(
                argv,
                cwd=root,
                check=False,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
        except OSError as exc:
            print(f"ERROR: failed to execute {command!r}: {exc}", file=sys.stderr)
            return 2
        if result.returncode != 0:
            print(
                f"ERROR: validation failed with exit code {result.returncode}: {command}",
                file=sys.stderr,
            )
            return result.returncode
    return 0


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
    quality_files = _changed_python_files(plan)
    if quality_files:
        lines.extend(
            [
                "",
                "## Changed-Python quality",
                "",
                f"- Base: `{plan.base_ref}`",
                (
                    "- `--run focused` and `--run full` first execute the "
                    "existing delta-aware Ruff, Black and mypy checkers."
                ),
                (
                    "- These reject newly introduced quality debt without "
                    "forcing unrelated legacy cleanup."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            *plan.commands,
            "```",
            "",
            (
                "> Use `--run focused` for changed-file quality plus the fast "
                "compile/test ring. Use `--run full` for the same quality gate "
                "plus the complete suite before merge."
            ),
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
    parser.add_argument(
        "--run",
        choices=("focused", "full"),
        help="Execute the generated focused ring or the focused ring plus full suite.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not (root / "src").exists() or not (root / "tests").exists():
        print(f"ERROR: {root} is not the repository root", file=sys.stderr)
        return 2
    plan = build(
        root,
        args.files or changed_from_git(root, args.base_ref),
        base_ref=args.base_ref,
    )
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
    if args.run:
        return run_plan(root, plan, args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
