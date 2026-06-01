"""Guardrails for logging ``extra`` keys.

Python logging rejects ``extra`` dictionaries that overwrite built-in
``LogRecord`` attributes.  This test catches obvious literal keys so startup
paths cannot regress with failures like ``Attempt to overwrite 'module'``.
"""

from __future__ import annotations

import ast
from pathlib import Path


RESERVED_LOG_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


def _literal_extra_reserved_keys(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                continue
            for key in keyword.value.keys:
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in RESERVED_LOG_RECORD_KEYS
                ):
                    findings.append((getattr(key, "lineno", node.lineno), key.value))
    return findings


def test_logging_extra_does_not_use_reserved_logrecord_keys() -> None:
    failures: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        for lineno, key in _literal_extra_reserved_keys(path):
            failures.append(f"{path}:{lineno} uses reserved logging extra key {key!r}")

    assert not failures, "\n".join(failures)
