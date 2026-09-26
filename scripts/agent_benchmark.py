#!/usr/bin/env python3
"""Validate or run the historical regression benchmark for coding-agent changes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("benchmarks/agent/historical_regressions.json")


def load_manifest(root: Path, path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_manifest(root: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return ["manifest requires a non-empty cases list"]

    seen: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            errors.append(f"{prefix} requires id")
        elif case_id in seen:
            errors.append(f"duplicate case id: {case_id}")
        seen.add(case_id)

        for field in ("category", "symptom", "invariant", "skill"):
            if not str(case.get(field) or "").strip():
                errors.append(f"{case_id or prefix} requires {field}")

        targets = case.get("pytest_targets")
        if not isinstance(targets, list) or not targets:
            errors.append(f"{case_id or prefix} requires pytest_targets")
            continue
        for target in targets:
            test_path = str(target).split("::", 1)[0]
            if not test_path.startswith("tests/"):
                errors.append(f"{case_id}: target outside tests/: {target}")
                continue
            if not (root / test_path).exists():
                errors.append(f"{case_id}: missing target {target}")
    return errors


def selected_targets(
    payload: dict[str, Any],
    case_ids: set[str] | None = None,
) -> list[str]:
    targets: list[str] = []
    for case in payload["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        targets.extend(str(item) for item in case["pytest_targets"])
    return list(dict.fromkeys(targets))


def run_benchmark(root: Path, targets: list[str]) -> int:
    if not targets:
        print("No benchmark targets selected.")
        return 2
    argv = [sys.executable, "-m", "pytest", "-q", *targets]
    print("+ " + " ".join(argv), file=sys.stderr)
    return subprocess.run(argv, cwd=root, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    payload = load_manifest(root, args.manifest)
    errors = validate_manifest(root, payload)
    case_ids = set(args.case)
    known_ids = {
        str(case["id"])
        for case in payload.get("cases", [])
        if isinstance(case, dict)
    }
    unknown = sorted(case_ids - known_ids)
    errors.extend(f"unknown case id: {item}" for item in unknown)

    targets = selected_targets(payload, case_ids or None) if not errors else []
    summary = {
        "valid": not errors,
        "errors": errors,
        "case_count": len(payload.get("cases", [])),
        "selected_cases": sorted(case_ids) if case_ids else "all",
        "pytest_targets": targets,
    }

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    elif args.list or args.validate or not args.run:
        print(f"Historical regression benchmark: {summary['case_count']} cases")
        for case in payload.get("cases", []):
            if case_ids and case.get("id") not in case_ids:
                continue
            print(f"- {case['id']}: {case['invariant']}")
        for error in errors:
            print(f"ERROR: {error}")

    if errors:
        return 1
    if args.run:
        return run_benchmark(root, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
