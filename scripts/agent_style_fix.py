#!/usr/bin/env python3
"""Safely auto-fix style on changed Python without expanding legacy debt."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

QUALITY_PREFIXES = ("src/", "dashboard/", "scripts/", "tests/")


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        check=False,
    )


def changed_python_files(
    root: Path,
    *,
    base_ref: str,
    explicit: Sequence[str] = (),
) -> tuple[str, ...]:
    if explicit:
        candidates = explicit
    else:
        result = _run(
            [
                "git",
                "diff",
                "--diff-filter=ACMR",
                "--name-only",
                base_ref,
                "--",
                "*.py",
            ],
            cwd=root,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git diff failed")
        candidates = result.stdout.splitlines()

    return tuple(
        dict.fromkeys(
            path
            for path in candidates
            if path.endswith(".py") and path.startswith(QUALITY_PREFIXES)
        )
    )


def _base_content(root: Path, base_ref: str, path: str) -> str | None:
    result = _run(
        ["git", "show", f"{base_ref}:{path}"],
        cwd=root,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _snapshot_clean(
    root: Path,
    *,
    content: str,
    tool: str,
) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snapshot.py"
        path.write_text(content, encoding="utf-8")
        if tool == "ruff":
            argv = [sys.executable, "-m", "ruff", "check", str(path)]
        elif tool == "black":
            argv = [sys.executable, "-m", "black", "--check", str(path)]
        else:
            raise ValueError(f"unsupported style tool: {tool}")
        return _run(argv, cwd=root).returncode == 0


def _fix_file(root: Path, *, base_ref: str, path: str) -> tuple[str, ...]:
    base_content = _base_content(root, base_ref, path)
    is_new = base_content is None
    actions: list[str] = []

    ruff_safe = is_new or _snapshot_clean(
        root,
        content=base_content,
        tool="ruff",
    )
    if ruff_safe:
        result = _run(
            [sys.executable, "-m", "ruff", "check", "--fix", path],
            cwd=root,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"Ruff failed for {path}")
        actions.append("ruff")
    else:
        actions.append("ruff-skipped-legacy-debt")

    black_safe = is_new or _snapshot_clean(
        root,
        content=base_content,
        tool="black",
    )
    if black_safe:
        result = _run(
            [sys.executable, "-m", "black", path],
            cwd=root,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Black failed for {path}")
        actions.append("black")
    else:
        actions.append("black-skipped-legacy-debt")

    return tuple(actions)


def _verify_delta_quality(
    root: Path,
    *,
    base_ref: str,
    files: Sequence[str],
) -> int:
    if not files:
        return 0
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="agent-style-files-",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("\n".join(files) + "\n")
        manifest = Path(handle.name)

    try:
        for checker in (
            "scripts/check_changed_ruff.py",
            "scripts/check_changed_black.py",
        ):
            path = root / checker
            if not path.exists():
                continue
            result = _run(
                [
                    sys.executable,
                    str(path),
                    "--base",
                    base_ref,
                    "--files-from",
                    str(manifest),
                ],
                cwd=root,
            )
            if result.returncode != 0:
                return result.returncode
    finally:
        manifest.unlink(missing_ok=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args(argv)

    root = args.root.resolve()
    files = changed_python_files(
        root,
        base_ref=args.base_ref,
        explicit=args.files,
    )
    if not files:
        print("PASS style preflight: no changed Python files")
        return 0

    for path in files:
        actions = _fix_file(root, base_ref=args.base_ref, path=path)
        print(f"{path}: {', '.join(actions)}")

    result = _verify_delta_quality(
        root,
        base_ref=args.base_ref,
        files=files,
    )
    if result == 0:
        print("PASS style preflight")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
