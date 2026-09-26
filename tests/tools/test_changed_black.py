from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path("scripts/check_changed_black.py")
_SPEC = importlib.util.spec_from_file_location("check_changed_black", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

changed_line_ranges = _MODULE.changed_line_ranges
formatting_spans = _MODULE.formatting_spans
ranges_intersect = _MODULE.ranges_intersect


def test_changed_line_ranges_parse_zero_context_hunks() -> None:
    diff = """@@ -10,2 +10,3 @@
@@ -30 +31,0 @@
@@ -40,0 +42,2 @@
"""
    assert changed_line_ranges(diff) == [(10, 12), (42, 43)]


def test_formatting_spans_identify_modified_source_lines() -> None:
    original = "a = 1\nb  = 2\nc = 3\n"
    formatted = "a = 1\nb = 2\nc = 3\n"
    assert formatting_spans(original, formatted) == [(2, 2)]


def test_range_intersection_distinguishes_legacy_from_new_debt() -> None:
    assert ranges_intersect([(20, 25)], [(23, 23)]) is True
    assert ranges_intersect([(1, 5)], [(10, 12)]) is False


def test_legacy_black_debt_outside_changed_lines_passes(
    tmp_path: Path, monkeypatch
) -> None:
    current = tmp_path / "legacy.py"
    current.write_text("bad  = 1\nkept = 2\nchanged = 3\n", encoding="utf-8")
    base = tmp_path / "base.py"
    base.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(_MODULE, "_base_file", lambda _base, _path: base)
    monkeypatch.setattr(_MODULE, "_black_check", lambda _path: False)
    monkeypatch.setattr(
        _MODULE,
        "_black_formatted_text",
        lambda _path: "bad = 1\nkept = 2\nchanged = 3\n",
    )
    monkeypatch.setattr(_MODULE, "_git_changed_ranges", lambda _base, _path: [(3, 3)])

    ok, message = _MODULE.check_file("origin/main", current)

    assert ok is True
    assert "only pre-existing Black debt remains" in message


def test_black_debt_on_changed_lines_fails(tmp_path: Path, monkeypatch) -> None:
    current = tmp_path / "legacy.py"
    current.write_text("kept = 1\nbad  = 2\n", encoding="utf-8")
    base = tmp_path / "base.py"
    base.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(_MODULE, "_base_file", lambda _base, _path: base)
    monkeypatch.setattr(_MODULE, "_black_check", lambda _path: False)
    monkeypatch.setattr(
        _MODULE,
        "_black_formatted_text",
        lambda _path: "kept = 1\nbad = 2\n",
    )
    monkeypatch.setattr(_MODULE, "_git_changed_ranges", lambda _base, _path: [(2, 2)])

    ok, message = _MODULE.check_file("origin/main", current)

    assert ok is False
    assert "Black would modify newly changed lines" in message


def test_git_changed_ranges_compares_base_to_worktree(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Result:
        returncode = 0
        stdout = "@@ -1 +1 @@\n"
        stderr = ""

    def fake_run(*args: str, **_kwargs):
        calls.append(args)
        return Result()

    monkeypatch.setattr(_MODULE, "_run", fake_run)

    assert _MODULE._git_changed_ranges("origin/main", Path("sample.py")) == [(1, 1)]
    assert calls
    assert "origin/main" in calls[0]
    assert "HEAD" not in calls[0]

