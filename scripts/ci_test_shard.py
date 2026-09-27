#!/usr/bin/env python3
"""Deterministically partition repository test files for parallel CI shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

DEFAULT_SHARD_COUNT = 4


def discover_test_files(root: Path) -> tuple[str, ...]:
    tests_root = root / "tests"
    if not tests_root.exists():
        return ()

    files = {
        path.relative_to(root).as_posix()
        for path in tests_root.rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    }
    return tuple(sorted(files))


def shard_files(
    files: Sequence[str],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[str, ...]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count - 1}], got {shard_index}"
        )

    ordered = tuple(sorted(dict.fromkeys(files)))
    return ordered[shard_index::shard_count]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=DEFAULT_SHARD_COUNT,
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    files = discover_test_files(root)
    if not files:
        parser.error("no pytest test files discovered")

    selected = shard_files(
        files,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not selected:
        parser.error(
            f"shard {args.shard_index}/{args.shard_count} contains no test files"
        )

    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
