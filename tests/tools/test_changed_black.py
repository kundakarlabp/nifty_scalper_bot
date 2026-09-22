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
