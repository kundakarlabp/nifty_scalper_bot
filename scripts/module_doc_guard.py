#!/usr/bin/env python3
"""Check key module summaries and prevent interface/documentation drift in CI.

Reads the architecture manifest; never imports production modules or contacts a broker.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


MANIFEST = Path("docs/architecture/agent_manifest.json")
ARCHITECTURE_DOCS = {
    "ARCHITECTURE_TRADING_PATH.md",
    "docs/TRADING_LIFECYCLE.md",
}


def covered_modules(payload: dict[str, object]) -> set[str]:
    """Resolve source paths from the existing architecture ownership SSOT."""
    paths = set(payload.get("runtime_path", []))
    for owner in payload.get("owners", []):
        if not isinstance(owner, dict):
            continue
        for key in ("owner", "read_facade"):
            value = owner.get(key)
            if isinstance(value, str) and value.endswith(".py"):
                paths.add(value)
        for value in owner.get("collaborators", []):
            if isinstance(value, str) and value.endswith(".py"):
                paths.add(value)
    paths.update(payload.get("module_documentation", []))
    return {path for path in paths if path.startswith("src/") and path.endswith(".py")}


def module_doc(source: str) -> str:
    """Read a module-level docstring without importing or executing the module."""
    return ast.get_docstring(ast.parse(source)) or ""


def api_inventory(source: str) -> tuple[tuple[str, ...], ...]:
    """Track callable names/signatures, including class methods, not their bodies."""
    tree = ast.parse(source)
    result: list[tuple[str, ...]] = []
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for item in tree.body:
        if isinstance(item, function_types):
            result.append(
                ("function", item.name, ast.dump(item.args), ast.dump(item.returns))
            )
        elif isinstance(item, ast.ClassDef):
            result.append(("class", item.name, ast.dump(ast.Tuple(elts=item.bases))))
            for method in item.body:
                if isinstance(method, function_types):
                    result.append(
                        (
                            "method",
                            item.name,
                            method.name,
                            ast.dump(method.args),
                            ast.dump(method.returns),
                        )
                    )
    return tuple(sorted(result))


def header_issue(path: str, source: str, *, minimum_chars: int = 100) -> str | None:
    """Require a short informative summary, not a bare module name."""
    try:
        header = module_doc(source)
    except SyntaxError as exc:
        return f"{path}: cannot parse module header: {exc}"
    if len(header.strip()) < minimum_chars or "\n" not in header:
        return f"{path}: opening module docstring needs a meaningful multi-line summary"
    markers = (
        "responsibilit",
        "runtime role",
        "owns",
        "boundary",
        "boundaries",
        "safety",
        "operational constraints",
        "position in the pipeline",
        "readiness helpers",
        "purpose:",
    )
    if not any(marker in header.lower() for marker in markers):
        return f"{path}: describe responsibility, ownership or safety in the header"
    return None


def git_read(root: Path, ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def git_changed(root: Path, base: str) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot compare with {base}: {result.stderr.strip()}")
    return set(result.stdout.splitlines())


def runtime_contract(payload: dict[str, object]) -> object:
    """Only ownership/path changes mandate an architecture documentation diff."""
    return payload.get("runtime_path"), payload.get("owners")


def check(root: Path, *, base: str | None = None) -> list[str]:
    payload = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    tracked = covered_modules(payload)
    problems: list[str] = []
    for path in sorted(tracked):
        file = root / path
        if not file.is_file():
            problems.append(f"{path}: manifest refers to a missing key module")
            continue
        problem = header_issue(path, file.read_text(encoding="utf-8"))
        if problem:
            problems.append(problem)

    if base is None:
        return problems

    changed = git_changed(root, base)
    for path in sorted(changed):
        if not (path.startswith("src/nifty_scalper_bot/") and path.endswith(".py")):
            continue
        file = root / path
        if not file.is_file():
            continue
        after = file.read_text(encoding="utf-8")
        before = git_read(root, base, path)
        if before is None:
            problem = header_issue(path, after)
            if problem and path not in tracked:
                problems.append(problem)
            continue
        if path not in tracked:
            continue
        try:
            changed_api = api_inventory(before) != api_inventory(after)
            unchanged_header = module_doc(before) == module_doc(after)
        except SyntaxError as exc:
            problems.append(f"{path}: cannot compare callable API: {exc}")
            continue
        if changed_api and unchanged_header:
            problems.append(
                f"{path}: callable interface changed; update the module header "
                "to describe/review the changed entrypoints"
            )

    if str(MANIFEST) in changed:
        previous = git_read(root, base, str(MANIFEST))
        if previous is not None:
            old = json.loads(previous)
            if runtime_contract(old) != runtime_contract(payload):
                if not changed.intersection(ARCHITECTURE_DOCS):
                    problems.append(
                        "Architecture owner/runtime-path changed: also review and "
                        "edit ARCHITECTURE_TRADING_PATH.md or docs/TRADING_LIFECYCLE.md"
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Check header coverage only")
    mode.add_argument("--base-ref", help="Check coverage plus change-specific drift")
    args = parser.parse_args(argv)
    try:
        issues = check(args.root.resolve(), base=args.base_ref)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Module documentation guard could not run: {exc}", file=sys.stderr)
        return 2
    if issues:
        for issue in issues:
            print(f"MODULE_DOC: {issue}", file=sys.stderr)
        return 1
    print("Module documentation: coverage and change-drift checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
