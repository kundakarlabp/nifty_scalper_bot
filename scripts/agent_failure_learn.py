#!/usr/bin/env python3
"""Classify CI/tool logs into known engineering-memory patterns without editing docs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FMT-001", re.compile(r"black|would reformat|formatting debt", re.I)),
    ("LINT-001", re.compile(r"ruff|\bE\d{3}\b|\bF\d{3}\b|\bI\d{3}\b", re.I)),
    ("TYPE-001", re.compile(r"mypy|error: .*\[[a-z-]+\]", re.I)),
    ("SYNTAX-001", re.compile(r"syntaxerror|indentationerror|compileall", re.I)),
    (
        "TEST-002",
        re.compile(
            r"flaky|timing|timeout|stale.*tick|passes? alone|order-sensitive",
            re.I,
        ),
    ),
    (
        "GIT-001",
        re.compile(r"stale.*base|main advanced|non-mergeable|head.*moved", re.I),
    ),
)


def classify(lines: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        for pattern_id, regex in PATTERNS:
            if regex.search(line):
                counts[pattern_id] += 1
    return counts


def render(counts: Counter[str], threshold: int) -> dict[str, object]:
    candidates = [
        {"pattern_id": pattern_id, "matches": count}
        for pattern_id, count in counts.most_common()
        if count >= threshold
    ]
    return {
        "known_pattern_matches": dict(counts),
        "candidate_repeats": candidates,
        "threshold": threshold,
        "policy": (
            "Candidates require human/root-cause review; this tool never edits "
            "ENGINEERING_FAILURE_PATTERNS.md automatically."
        ),
    }


def _read_inputs(paths: list[Path]) -> list[str]:
    if not paths:
        return sys.stdin.read().splitlines()
    lines: list[str] = []
    for path in paths:
        lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    args = parser.parse_args()

    payload = render(classify(_read_inputs(args.paths)), max(1, args.threshold))
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print("Engineering-memory candidates")
        for item in payload["candidate_repeats"]:
            print(f"- {item['pattern_id']}: {item['matches']} matching lines")
        if not payload["candidate_repeats"]:
            print("- none")
        print(payload["policy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
