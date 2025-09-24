"""Utilities for running lint/type/test checks on a single file.

This module exposes :func:`run_file_diagnostics` which executes Ruff, mypy
and pytest (for test files) on a given path. It returns a human‑readable
string summarizing the results so it can be surfaced via Telegram.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[str, int]:
    """Run *cmd* and return its combined output and return code."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    return out, proc.returncode


def run_file_diagnostics(path: str) -> str:
    """Run Ruff, mypy and pytest (when applicable) on *path*.

    Parameters
    ----------
    path:
        File path relative to the repository root or absolute.

    Returns
    -------
    str
        Summary of results from each tool. ``OK`` indicates success; otherwise
        the captured tool output is returned so the caller can inspect the
        problem.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return f"File not found: {path}"

    cmds: list[tuple[str, list[str]]] = [
        ("ruff", ["ruff", str(p)]),
        ("mypy", ["mypy", str(p)]),
    ]
    if p.suffix == ".py" and (p.name.startswith("test") or "tests" in p.parts):
        cmds.append(("pytest", ["pytest", str(p)]))

    results: list[str] = []
    for name, cmd in cmds:
        out, code = _run(cmd)
        if code == 0:
            results.append(f"{name}: OK")
        else:
            results.append(f"{name} error:\n{out}")
    return "\n\n".join(results)
