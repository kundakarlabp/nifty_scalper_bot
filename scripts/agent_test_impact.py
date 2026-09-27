#!/usr/bin/env python3
"""Select exact tests that directly import or co-locate with changed modules."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def module_name(path: str) -> str | None:
    prefix = "src/"
    if not path.startswith(prefix) or not path.endswith(".py"):
        return None
    value = path[len(prefix) : -3].replace("/", ".")
    if value.endswith(".__init__"):
        value = value[: -len(".__init__")]
    return value or None


def _import_needles(module: str) -> tuple[str, ...]:
    parent, _, leaf = module.rpartition(".")
    needles = {
        module,
        f"from {module} import ",
        f"import {module}",
    }
    if parent and leaf:
        needles.add(f"from {parent} import {leaf}")
    return tuple(sorted(needles))


def _co_located(path: Path, source_stems: set[str]) -> bool:
    stem = path.stem.lower()
    if stem.startswith("test_"):
        stem = stem[5:]
    return any(
        stem == source
        or stem.startswith(f"{source}_")
        or source.startswith(f"{stem}_")
        for source in source_stems
    )


def impacted_tests(
    root: Path,
    files: Sequence[str],
    *,
    limit: int = 32,
) -> tuple[str, ...]:
    changed_tests = [
        path
        for path in files
        if path.startswith("tests/") and path.endswith(".py") and (root / path).exists()
    ]
    modules = tuple(
        module
        for path in files
        if (module := module_name(path)) is not None
    )
    source_stems = {
        Path(path).stem.lower()
        for path in files
        if path.startswith("src/") and path.endswith(".py")
    }
    if not modules and not changed_tests:
        return ()

    needles = tuple(
        needle
        for module in modules
        for needle in _import_needles(module)
    )
    matches: list[str] = list(changed_tests)

    tests_root = root / "tests"
    if tests_root.exists():
        for path in sorted(tests_root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel in matches:
                continue
            if _co_located(path, source_stems):
                matches.append(rel)
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(needle in text for needle in needles):
                matches.append(rel)
            if len(matches) >= limit:
                break

    return tuple(dict.fromkeys(matches))[:limit]
