from __future__ import annotations

from scripts.check_changed_mypy import errors


def test_error_comparison_ignores_line_shifts_but_rejects_new_errors() -> None:
    before = errors("app.py:10: error: name defined on line 4  [no-redef]\n")
    after = errors(
        "app.py:20: error: name defined on line 14  [no-redef]\n"
        "app.py:21: error: name defined on line 14  [no-redef]\n"
    )

    assert sum((after - before).values()) == 1
