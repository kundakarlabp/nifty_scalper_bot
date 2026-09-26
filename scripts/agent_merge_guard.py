#!/usr/bin/env python3
"""Fail closed when the branch being merged is not the exact validated base/head."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _rev_parse(root: Path, ref: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"cannot resolve {ref}")
    return result.stdout.strip()


def verify(
    root: Path,
    *,
    validated_base: str,
    validated_head: str,
    base_ref: str,
    head_ref: str,
) -> dict[str, object]:
    current_base = _rev_parse(root, base_ref)
    current_head = _rev_parse(root, head_ref)
    merge_base = _rev_parse(root, f"{base_ref}^{{commit}}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", current_base, current_head],
        cwd=root,
        check=False,
    ).returncode == 0

    checks = {
        "base_matches": current_base == validated_base,
        "head_matches": current_head == validated_head,
        "base_is_ancestor": ancestry,
        "base_ref_resolves": merge_base == current_base,
    }
    return {
        "ok": all(checks.values()),
        "validated_base": validated_base,
        "validated_head": validated_head,
        "current_base": current_base,
        "current_head": current_head,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--validated-base", required=True)
    parser.add_argument("--validated-head", required=True)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    try:
        result = verify(
            args.root.resolve(),
            validated_base=args.validated_base,
            validated_head=args.validated_head,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
        )
    except RuntimeError as exc:
        print(f"FAIL merge guard: {exc}")
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2))
    elif result["ok"]:
        print(
            "PASS merge guard "
            f"base={result['current_base']} head={result['current_head']}"
        )
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
