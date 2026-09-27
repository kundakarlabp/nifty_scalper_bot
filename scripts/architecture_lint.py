#!/usr/bin/env python3
"""Check mechanically provable ownership rules from the architecture manifest."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scripts.agent_architecture import architecture_rules, load_manifest
except ModuleNotFoundError:
    from agent_architecture import architecture_rules, load_manifest


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    detail: str


def _call_name(node: ast.Call) -> str | None:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _violates_rule(
    *,
    rel: str,
    call_name: str,
    rule: dict[str, Any],
) -> bool:
    calls = {str(item) for item in rule.get("calls", [])}
    if call_name not in calls:
        return False

    kind = rule.get("kind")
    if kind == "call_allowlist":
        allowlist = {str(item) for item in rule.get("allowlist", [])}
        return rel not in allowlist

    if kind == "forbidden_calls_under":
        prefixes = tuple(str(item) for item in rule.get("path_prefixes", []))
        return any(rel.startswith(prefix) for prefix in prefixes)

    raise ValueError(f"unsupported architecture rule kind: {kind!r}")


def inspect_file(
    root: Path,
    path: Path,
    *,
    rules: Sequence[dict[str, Any]] | None = None,
) -> list[Violation]:
    rel = path.relative_to(root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [
            Violation(
                "python-syntax",
                rel,
                int(exc.lineno or 0),
                str(exc.msg),
            )
        ]

    active_rules = (
        tuple(rules)
        if rules is not None
        else architecture_rules(load_manifest(root))
    )
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name is None:
            continue
        for rule in active_rules:
            if not _violates_rule(rel=rel, call_name=call_name, rule=rule):
                continue
            violations.append(
                Violation(
                    str(rule["id"]),
                    rel,
                    node.lineno,
                    f"forbidden owner call {call_name}",
                )
            )
    return violations


def inspect_repository(root: Path) -> list[Violation]:
    src = root / "src" / "nifty_scalper_bot"
    rules = architecture_rules(load_manifest(root))
    violations: list[Violation] = []
    for path in sorted(src.rglob("*.py")):
        violations.extend(inspect_file(root, path, rules=rules))
    return violations


def _format_text(violations: Iterable[Violation]) -> str:
    return "\n".join(
        f"{item.rule}: {item.path}:{item.line}: {item.detail}"
        for item in violations
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    root = args.root.resolve()
    violations = inspect_repository(root)
    if args.format == "json":
        print(json.dumps([item.__dict__ for item in violations], indent=2))
    elif violations:
        print(_format_text(violations))
    else:
        print("PASS architecture ownership lint")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
