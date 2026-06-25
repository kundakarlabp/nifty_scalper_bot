"""Run pytest while emitting and persisting its final diagnostic block."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys

import pytest


def main() -> int:
    buffer = io.StringIO()
    arguments = ["-q", "-x", "--tb=short", *sys.argv[1:]]
    with redirect_stdout(buffer), redirect_stderr(buffer):
        result = pytest.main(arguments)
    lines = buffer.getvalue().splitlines()
    diagnostic = "\n".join(lines[-200:]) + "\n"
    output_path = Path(os.getenv("PYTEST_COMPACT_LOG", "pytest-compact.log"))
    output_path.write_text(diagnostic, encoding="utf-8")
    print(diagnostic, end="")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
