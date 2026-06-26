#!/usr/bin/env python3
"""File purpose: Build a compact repository context report for coding agents.
Key responsibilities: Rank relevant files and Python symbols, identify related tests, and suggest validation commands.
Operational constraints: Skip environment/runtime data and emit signatures rather than source implementations.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence

ALLOWED = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".sh"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", "data", "logs", "runtime", "artifacts",
    "backups", "dist", "htmlcov",
}
SKIP_NAMES = {".env", ".env.local", ".env.production", "credentials.json", "token.json"}
SKIP_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log", ".pem", ".key"}
STOP = {
    "about", "after", "again", "also", "been", "before", "bot", "code",
    "could", "error", "find", "from", "have", "into", "issue", "modify",
    "need", "please", "repo", "repository", "should", "that", "the", "then",
    "there", "this", "using", "want", "when", "where", "which", "with",
}
CANONICAL = {
    "src/nifty_scalper_bot/core/app.py",
    "src/nifty_scalper_bot/data/market_data_manager.py",
    "src/nifty_scalper_bot/data/data_hub.py",
    "src/nifty_scalper_bot/strategies/runner.py",
    "src/nifty_scalper_bot/execution/order_manager.py",
    "src/nifty_scalper_bot/execution/bracket_manager.py",
    "src/nifty_scalper_bot/notifications/telegram_controller.py",
}
AREA_TESTS = {
    "streaming": ("tests/streaming", "tests/data"),
    "data": ("tests/data", "tests/core"),
    "execution": ("tests/execution", "tests/integration", "tests/test_execution_path_contract.py"),
    "risk": ("tests/risk",),
    "strategies": ("tests/strategies",),
    "notifications": ("tests/notifications",),
    "dashboard": ("tests/dashboard",),
    "deploy": ("tests/test_deployment_release_guard.py", "tests/core/test_release_guard.py"),
    "core": ("tests/core", "tests/architecture"),
}


def terms_from(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_.:-]{2,}", query.lower())
    terms: set[str] = set()
    for token in raw:
        terms.add(token)
        terms.update(part for part in re.split(r"[_.:-]+", token) if len(part) >= 3)
    return sorted(term for term in terms if term not in STOP and len(term) < 40)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        rel = path.relative_to(root)
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.stat().st_size <= 1_500_000:
            yield path


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def python_symbols(path: str, text: str, terms: Sequence[str]) -> list[dict[str, object]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found: list[dict[str, object]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def add(self, node: ast.AST, name: str, signature: str) -> None:
            qualified = ".".join((*self.scope, name)) if self.scope else name
            score = sum(40 for term in terms if term in qualified.lower())
            found.append({
                "path": path,
                "line": int(getattr(node, "lineno", 1)),
                "name": qualified,
                "signature": signature,
                "score": score,
            })

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            bases = ", ".join(safe_unparse(base) for base in node.bases)
            self.add(node, node.name, f"class {node.name}{f'({bases})' if bases else ''}")
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function(node, "def")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function(node, "async def")

        def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str) -> None:
            returns = f" -> {safe_unparse(node.returns)}" if node.returns else ""
            self.add(node, node.name, f"{prefix} {node.name}({safe_unparse(node.args)}){returns}")
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    Visitor().visit(tree)
    return found


def recent_files(root: Path) -> Counter[str]:
    try:
        result = subprocess.run(
            ["git", "log", "-n", "25", "--name-only", "--pretty=format:"],
            cwd=root, capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return Counter()
    return Counter(line.strip() for line in result.stdout.splitlines() if line.strip())


def risk(path: str) -> str:
    lowered = f"/{path.lower()}"
    if any(part in lowered for part in ("/execution/", "/risk/", "/streaming/", "/core/app.py", "/data/rest/")):
        return "HIGH"
    if any(part in lowered for part in ("/strategies/", "/data/", "/config/", "/notifications/", "/deploy/")):
        return "MEDIUM"
    return "LOW"


def build(root: Path, query: str, max_files: int, max_symbols: int) -> dict[str, object]:
    terms = terms_from(query)
    recent = recent_files(root)
    records: list[dict[str, object]] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        symbols = python_symbols(rel, text, terms) if path.suffix == ".py" else []
        lowered_path, lowered_text = rel.lower(), text.lower()
        reasons: list[str] = []
        score = 10 if rel in CANONICAL else 0
        if rel in CANONICAL:
            reasons.append("canonical-runtime")
        for term in terms:
            if term in lowered_path:
                score += 35
                reasons.append(f"path:{term}")
            hits = min(lowered_text.count(term), 10)
            if hits:
                score += hits * 3
                reasons.append(f"text:{term}")
            if any(item["score"] for item in symbols if term in str(item["name"]).lower()):
                score += 30
                reasons.append(f"symbol:{term}")
        if recent[rel]:
            score += min(recent[rel] * 3, 15)
            reasons.append("recently-changed")
        records.append({"path": rel, "score": score, "risk": risk(rel), "reasons": sorted(set(reasons)), "symbols": symbols})

    ranked = sorted((item for item in records if item["score"] > 0), key=lambda item: (-int(item["score"]), str(item["path"])))[:max_files]
    symbols = sorted(
        (symbol for item in ranked for symbol in item["symbols"] if int(symbol["score"]) > 0),
        key=lambda item: (-int(item["score"]), str(item["path"]), int(item["line"])),
    )[:max_symbols]
    source_stems = {Path(str(item["path"])).stem.lower() for item in ranked if not str(item["path"]).startswith("tests/")}
    tests = []
    for item in records:
        path = str(item["path"])
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        stem = Path(path).stem.lower().removeprefix("test_")
        if stem in source_stems or int(item["score"]) > 0:
            tests.append(path)
    tests = list(dict.fromkeys(tests))[:12]

    commands = ["python -m compileall -q src dashboard"]
    if tests:
        commands.append("python -m pytest -q " + " ".join(tests[:8]))
    joined = " ".join(str(item["path"]).lower() for item in ranked)
    for area, candidates in AREA_TESTS.items():
        if area in joined:
            existing = [candidate for candidate in candidates if (root / candidate).exists()]
            if existing:
                commands.append("python -m pytest -q " + " ".join(existing))
    commands.append("python -m pytest -q  # mandatory before merge")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "terms": terms,
        "ranked_files": [{key: item[key] for key in ("path", "score", "risk", "reasons")} for item in ranked],
        "symbols": symbols,
        "related_tests": tests,
        "commands": list(dict.fromkeys(commands)),
    }


def markdown(data: dict[str, object]) -> str:
    lines = [
        "<!-- agent-context-report -->", "# Agent Context Report", "",
        f"Generated: `{data['generated_at']}`", f"Terms: `{', '.join(data['terms']) or 'none'}`", "",
        "## Repository contract", "",
        "- NIFTY options are the only tradable instruments; spot and futures are context only.",
        "- Preserve `core/app.py → market_data_manager.py → data_hub.py → strategies/runner.py → order_manager.py → bracket_manager.py → telegram_controller.py`.",
        "- Never bypass readiness, quote-quality, risk, capital, cooldown, position, max-loss, or execution-mode gates.",
        "- Environment files, runtime data, logs, databases, key material, and implementation bodies are excluded.", "",
        "## Ranked reading order", "", "| # | File | Score | Risk | Why |", "|---:|---|---:|---|---|",
    ]
    for index, item in enumerate(data["ranked_files"], 1):
        lines.append(f"| {index} | `{item['path']}` | {item['score']} | {item['risk']} | {', '.join(item['reasons'][:5]) or 'structural relevance'} |")
    if not data["ranked_files"]:
        lines.append("| 1 | _No strong match_ | 0 | — | Add an exact symbol, log message, or module |")
    lines.extend(["", "## Matching symbols", ""])
    lines.extend(f"- `{item['path']}:{item['line']}` — `{item['name']}` — `{item['signature']}`" for item in data["symbols"])
    if not data["symbols"]:
        lines.append("- No direct Python symbol match.")
    lines.extend(["", "## Related regression tests", ""])
    lines.extend(f"- `{path}`" for path in data["related_tests"])
    if not data["related_tests"]:
        lines.append("- Add a focused regression test through the owning public interface.")
    lines.extend(["", "## Suggested validation", "", "```bash", *data["commands"], "```", "", "## Execution order", "", "1. Read `AGENTS.md`, `docs/REPO_MAP.md`, and only the top-ranked files.", "2. Reproduce the symptom with a red-capable test or fixture replay.", "3. Establish root cause and untouched boundaries.", "4. Implement the smallest owner-consistent correction with regression coverage.", "5. Use final-head CI and resolved review threads as squash-merge evidence.", ""])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--query", default="")
    parser.add_argument("--query-file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--max-files", type=int, default=18)
    parser.add_argument("--max-symbols", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not (root / "src").exists() or not (root / "tests").exists():
        print(f"ERROR: {root} is not the repository root", file=sys.stderr)
        return 2
    query = args.query
    if args.query_file:
        query = f"{query}\n{Path(args.query_file).read_text(encoding='utf-8', errors='replace')}"
    data = build(root, query or "runtime readiness execution", max(args.max_files, 1), max(args.max_symbols, 1))
    output = json.dumps(data, indent=2, sort_keys=True) if args.format == "json" else markdown(data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output.rstrip() + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
