#!/usr/bin/env python3
"""Fail only on Black debt introduced by the current change.

Files that were Black-clean at the comparison base remain subject to the normal
full-file Black check. For a legacy file that was already Black-dirty at the
base, formatting differences are allowed only when they are wholly outside
lines changed by the current branch.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

_HUNK_RE = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<length>\d+))? @@"
)


def changed_line_ranges(diff_text: str) -> list[tuple[int, int]]:
    """Return inclusive current-file line ranges changed by a zero-context diff."""
    ranges: list[tuple[int, int]] = []
    for line in diff_text.splitlines():
        match = _HUNK_RE.match(line)
        if match is None:
            continue
        start = int(match.group("start"))
        length_text = match.group("length")
        length = 1 if length_text is None else int(length_text)
        if length > 0:
            ranges.append((start, start + length - 1))
    return ranges


def formatting_spans(original: str, formatted: str) -> list[tuple[int, int]]:
    """Return source-line spans that Black would modify."""
    if original == formatted:
        return []
    original_lines = original.splitlines(keepends=True)
    formatted_lines = formatted.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=original_lines, b=formatted_lines)
    spans: list[tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            spans.append((i1 + 1, i2))
        else:
            point = max(1, i1 + 1)
            spans.append((point, point))
    return spans


def ranges_intersect(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> bool:
    """Return whether any inclusive line ranges overlap."""
    return any(
        max(left_start, right_start) <= min(left_end, right_end)
        for left_start, left_end in left
        for right_start, right_end in right
    )


def _run(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        capture_output=True,
    )


def _black_check(path: Path) -> bool:
    result = _run(
        sys.executable,
        "-m",
        "black",
        "--check",
        "--quiet",
        "--config",
        "pyproject.toml",
        str(path),
    )
    return result.returncode == 0


def _black_formatted_text(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="changed-black-") as tmp_dir:
        candidate = Path(tmp_dir) / path.name
        candidate.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        result = _run(
            sys.executable,
            "-m",
            "black",
            "--quiet",
            "--config",
            "pyproject.toml",
            str(candidate),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Black failed while formatting {path}: {result.stderr.strip()}"
            )
        return candidate.read_text(encoding="utf-8")


def _base_file(base: str, path: Path) -> Path | None:
    result = _run("git", "show", f"{base}:{path.as_posix()}")
    if result.returncode != 0:
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=path.suffix,
        prefix="base-black-",
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        tmp.write(result.stdout)
    return Path(tmp.name)


def _git_changed_ranges(base: str, path: Path) -> list[tuple[int, int]]:
    result = _run(
        "git",
        "diff",
        "--unified=0",
        "--diff-filter=ACMR",
        f"{base}...HEAD",
        "--",
        path.as_posix(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git diff failed for {path}: {result.stderr.strip()}"
        )
    return changed_line_ranges(result.stdout)


def check_file(base: str, path: Path) -> tuple[bool, str]:
    """Check one changed Python file against Black without inheriting old debt."""
    if not path.exists():
        return True, f"SKIP {path}: file no longer exists"

    base_path = _base_file(base, path)
    if base_path is None:
        clean = _black_check(path)
        return clean, (
            f"PASS {path}: new file is Black-clean"
            if clean
            else f"FAIL {path}: new file is not Black-clean"
        )

    try:
        if _black_check(base_path):
            clean = _black_check(path)
            return clean, (
                f"PASS {path}: Black-clean file remains clean"
                if clean
                else f"FAIL {path}: Black-clean base gained formatting debt"
            )

        original = path.read_text(encoding="utf-8")
        formatted = _black_formatted_text(path)
        black_spans = formatting_spans(original, formatted)
        changed_spans = _git_changed_ranges(base, path)

        if not black_spans:
            return True, f"PASS {path}: legacy file is now Black-clean"
        if not changed_spans:
            return (
                False,
                f"FAIL {path}: changed file has no resolvable changed-line ranges",
            )
        if ranges_intersect(black_spans, changed_spans):
            return (
                False,
                f"FAIL {path}: Black would modify newly changed lines "
                f"(changed={changed_spans}, black={black_spans})",
            )
        return (
            True,
            f"PASS {path}: only pre-existing Black debt remains outside changed lines",
        )
    finally:
        base_path.unlink(missing_ok=True)


def _read_paths(files_from: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in files_from.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--files-from", type=Path, required=True)
    args = parser.parse_args()

    failed = False
    for path in _read_paths(args.files_from):
        ok, message = check_file(args.base, path)
        print(message)
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
