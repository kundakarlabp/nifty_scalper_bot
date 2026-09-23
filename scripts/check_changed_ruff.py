#!/usr/bin/env python3
"""Fail only on Ruff debt introduced by the current change.

Changed legacy files may retain diagnostics that already existed at the
comparison base. Every new diagnostic fingerprint remains a hard failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DiagnosticFingerprint = tuple[str, str, str]


def _run(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_file(base: str, path: Path) -> str | None:
    result = _run("git", "show", f"{base}:{path.as_posix()}")
    if result.returncode != 0:
        return None
    return result.stdout


def _ruff_diagnostics(path: Path, content: str) -> list[dict[str, Any]]:
    result = _run(
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--output-format=json",
        "--stdin-filename",
        path.as_posix(),
        "-",
        input_text=content,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(
            f"Ruff failed for {path}: {(result.stderr or result.stdout).strip()}"
        )
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Ruff output for {path}")
    return [item for item in payload if isinstance(item, dict)]


def diagnostic_fingerprint(
    diagnostic: dict[str, Any],
    source_lines: list[str],
) -> DiagnosticFingerprint:
    """Return a line-stable identity for one Ruff diagnostic."""
    code = str(diagnostic.get("code") or "")
    message = str(diagnostic.get("message") or "")
    location = diagnostic.get("location") or {}
    try:
        row = int(location.get("row") or 0)
    except (TypeError, ValueError):
        row = 0
    source = source_lines[row - 1].strip() if 0 < row <= len(source_lines) else ""
    return code, message, source


def diagnostic_counts(content: str, diagnostics: list[dict[str, Any]]) -> Counter:
    """Count Ruff diagnostics using fingerprints stable across line shifts."""
    lines = content.splitlines()
    return Counter(diagnostic_fingerprint(item, lines) for item in diagnostics)


def introduced_diagnostics(
    base_content: str | None,
    current_content: str,
    *,
    path: Path,
) -> Counter:
    """Return diagnostics present beyond the comparison-base multiset."""
    current = diagnostic_counts(
        current_content,
        _ruff_diagnostics(path, current_content),
    )
    if base_content is None:
        return current
    base = diagnostic_counts(
        base_content,
        _ruff_diagnostics(path, base_content),
    )
    return current - base


def check_file(base: str, path: Path) -> tuple[bool, str]:
    """Check one file without inheriting Ruff debt already present at base."""
    if not path.exists():
        return True, f"SKIP {path}: file no longer exists"

    current_content = path.read_text(encoding="utf-8")
    base_content = _git_file(base, path)
    introduced = introduced_diagnostics(
        base_content,
        current_content,
        path=path,
    )
    if not introduced:
        return True, f"PASS {path}: no new Ruff diagnostics"

    for (code, message, source), count in introduced.items():
        suffix = f" x{count}" if count > 1 else ""
        print(f"{path}: NEW {code} {message}{suffix}")
        if source:
            print(f"    {source}")
    count = sum(int(value) for value in introduced.values())
    return False, f"FAIL {path}: introduced {count} Ruff diagnostic(s)"


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
