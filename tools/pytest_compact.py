"""Run pytest while emitting only its final diagnostic block."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import sys

import pytest


def main() -> int:
    buffer = io.StringIO()
    arguments = ["-q", "-x", "--tb=short", *sys.argv[1:]]
    with redirect_stdout(buffer), redirect_stderr(buffer):
        result = pytest.main(arguments)
    lines = buffer.getvalue().splitlines()
    print("\n".join(lines[-120:]))
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
