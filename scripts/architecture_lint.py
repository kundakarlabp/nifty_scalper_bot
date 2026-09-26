#!/usr/bin/env python3
"""Check a small set of mechanically provable architecture ownership rules."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BROKER_HISTORY_ALLOWLIST = {
    "src/nifty_scalper_bot/data/market_data_manager.py",
    "src/nifty_scalper_bot/data/rest/zerodha_client.py",
}
HYDRATION_FORBIDDEN_PREFIXES = (
    "src/nifty_scalper_bot/execution/",
    "src/nifty_scalper_bot/notifications/",
)
HYDRATION_OWNER_CALLS = {
    "hydrate_symbol_history",
    "ensure_history",
    "reseed_history_from_bars",
    "ingest_historical_bar",
    "replace_history",
}
STRATEGY_FORBIDDEN_CALLS = {
    "historical_data",
    "get_historical_data",
    "fetch_history",
    "instruments",
}


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


def inspect_file(root: Path, path: Path) -> list[Violation]:
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

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue

        if (
            name in {"historical_data", "get_historical_data"}
            and rel not in BROKER_HISTORY_ALLOWLIST
        ):
            violations.append(
                Violation(
                    "broker-history-owner",
                    rel,
                    node.lineno,
                    f"unexpected call to {name}",
                )
            )

        if rel.startswith("src/nifty_scalper_bot/strategies/") and (
            name in STRATEGY_FORBIDDEN_CALLS
        ):
            violations.append(
                Violation(
                    "strategy-boundary",
                    rel,
                    node.lineno,
                    f"strategy calls forbidden owner API {name}",
                )
            )

        if any(rel.startswith(prefix) for prefix in HYDRATION_FORBIDDEN_PREFIXES) and (
            name in HYDRATION_OWNER_CALLS
        ):
            violations.append(
                Violation(
                    "history-hydration-owner",
                    rel,
                    node.lineno,
                    f"non-owner calls {name}",
                )
            )

    return violations


def inspect_repository(root: Path) -> list[Violation]:
    src = root / "src" / "nifty_scalper_bot"
    violations: list[Violation] = []
    for path in sorted(src.rglob("*.py")):
        violations.extend(inspect_file(root, path))
    return violations


def _format_text(violations: Iterable[Violation]) -> str:
    rows = [
        f"{item.rule}: {item.path}:{item.line}: {item.detail}" for item in violations
    ]
    return "\n".join(rows)


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
