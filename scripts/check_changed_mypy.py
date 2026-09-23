#!/usr/bin/env python3
"""Reject new mypy errors in changed production files with existing type debt."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

_ERROR = re.compile(r"^.+?:\d+: error: (.+)$", re.MULTILINE)


def errors(output: str) -> Counter[str]:
    """Ignore file locations and line shifts while retaining error multiplicity."""
    return Counter(
        re.sub(r"on line \d+", "on line <base>", item)
        for item in _ERROR.findall(output)
    )


def _mypy(path: Path) -> Counter[str]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--follow-imports=skip", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stdout + result.stderr)
    return errors(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--files-from", type=Path, required=True)
    args = parser.parse_args()
    failed = False
    for name in args.files_from.read_text(encoding="utf-8").splitlines():
        if not name.startswith("src/nifty_scalper_bot/") or not name.endswith(".py"):
            continue
        path = Path(name)
        current = _mypy(path)
        base = subprocess.run(
            ["git", "show", f"{args.base}:{name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        previous: Counter[str] = Counter()
        if base.returncode == 0:
            with tempfile.TemporaryDirectory() as tmp:
                snapshot = Path(tmp) / path.name
                snapshot.write_text(base.stdout, encoding="utf-8")
                previous = _mypy(snapshot)
        introduced = current - previous
        if introduced:
            failed = True
            for message, count in introduced.items():
                print(f"{name}: NEW {message} (x{count})")
        else:
            print(f"PASS {name}: no new mypy errors")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
