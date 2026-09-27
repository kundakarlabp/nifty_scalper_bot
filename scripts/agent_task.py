#!/usr/bin/env python3
"""Single public entry point for coding-agent context, validation, and merge checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _run(
    root: Path,
    argv: Sequence[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=root,
        capture_output=capture_output,
        text=True,
        check=False,
    )


def _python(root: Path, script: str, *args: str, capture_output: bool = False):
    return _run(
        root,
        [sys.executable, str(root / "scripts" / script), *args],
        capture_output=capture_output,
    )


def _file_args(files: Sequence[str]) -> list[str]:
    return ["--files", *files] if files else []


def _plan(
    root: Path,
    *,
    base_ref: str,
    files: Sequence[str],
) -> dict[str, object]:
    result = _python(
        root,
        "agent_check.py",
        "--base-ref",
        base_ref,
        *_file_args(files),
        "--format",
        "json",
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "agent_check plan failed")
    return json.loads(result.stdout)


def _run_benchmarks(
    root: Path,
    *,
    areas: Sequence[str],
) -> int:
    args: list[str] = []
    for area in areas:
        args.extend(["--area", area])

    preview = _python(
        root,
        "agent_benchmark.py",
        *args,
        "--format",
        "json",
        capture_output=True,
    )
    if preview.returncode != 0:
        sys.stderr.write(preview.stderr)
        sys.stdout.write(preview.stdout)
        return preview.returncode

    payload = json.loads(preview.stdout)
    targets = payload.get("pytest_targets") or []
    if not targets:
        print("No historical regression benchmark matches this change.")
        return 0

    selected = payload.get("selected_cases")
    print(f"Historical regression cases: {selected}")
    return _python(root, "agent_benchmark.py", *args, "--run").returncode


def _syntax_preflight(root: Path) -> int:
    """Fail before style mutation when repository Python cannot compile."""
    return _run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "dashboard",
            "scripts",
        ],
        cwd=root,
    ).returncode


def _validate(
    root: Path,
    *,
    base_ref: str,
    files: Sequence[str],
    scope: str,
    style_fix: bool,
) -> int:
    syntax = _syntax_preflight(root)
    if syntax != 0:
        return syntax

    if style_fix:
        style = _python(
            root,
            "agent_style_fix.py",
            "--base-ref",
            base_ref,
            *_file_args(files),
        )
        if style.returncode != 0:
            return style.returncode

    plan = _plan(root, base_ref=base_ref, files=files)
    validate = _python(
        root,
        "agent_check.py",
        "--base-ref",
        base_ref,
        *_file_args(files),
        "--run",
        scope,
    )
    if validate.returncode != 0:
        return validate.returncode

    areas = tuple(str(item) for item in plan.get("areas", []))
    return _run_benchmarks(root, areas=areas)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    context = subparsers.add_parser("context")
    context.add_argument("--query", required=True)
    context.add_argument("--output", type=Path)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--base-ref", default="origin/main")
    plan.add_argument("--files", nargs="*", default=[])
    plan.add_argument("--output", type=Path)
    plan.add_argument("--format", choices=("markdown", "json"), default="markdown")

    for name in ("check", "full"):
        command = subparsers.add_parser(name)
        command.add_argument("--base-ref", default="origin/main")
        command.add_argument("--files", nargs="*", default=[])
        command.add_argument("--no-style-fix", action="store_true")

    merge_check = subparsers.add_parser("merge-check")
    merge_check.add_argument("--validated-base", required=True)
    merge_check.add_argument("--validated-head", required=True)
    merge_check.add_argument("--base-ref", default="origin/main")
    merge_check.add_argument("--head-ref", default="HEAD")

    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.command == "context":
        context_args = ["--query", args.query]
        if args.output:
            context_args.extend(["--output", str(args.output)])
        return _python(root, "agent_context.py", *context_args).returncode

    if args.command == "plan":
        plan_args = [
            "--base-ref",
            args.base_ref,
            *_file_args(args.files),
            "--format",
            args.format,
        ]
        if args.output:
            plan_args.extend(["--output", str(args.output)])
        return _python(root, "agent_check.py", *plan_args).returncode

    if args.command in {"check", "full"}:
        return _validate(
            root,
            base_ref=args.base_ref,
            files=args.files,
            scope="focused" if args.command == "check" else "full",
            style_fix=not args.no_style_fix,
        )

    if args.command == "merge-check":
        return _python(
            root,
            "agent_merge_guard.py",
            "--validated-base",
            args.validated_base,
            "--validated-head",
            args.validated_head,
            "--base-ref",
            args.base_ref,
            "--head-ref",
            args.head_ref,
        ).returncode

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
