from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path("scripts/check_changed_ruff.py")
_SPEC = importlib.util.spec_from_file_location("check_changed_ruff", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

diagnostic_counts = _MODULE.diagnostic_counts
diagnostic_fingerprint = _MODULE.diagnostic_fingerprint


def _diag(code: str, message: str, row: int) -> dict:
    return {
        "code": code,
        "message": message,
        "location": {"row": row, "column": 1},
    }


def test_fingerprint_is_stable_across_line_number_shift() -> None:
    first = diagnostic_fingerprint(_diag("E501", "Line too long", 1), ["same = line"])
    shifted = diagnostic_fingerprint(
        _diag("E501", "Line too long", 3),
        ["x", "y", "same = line"],
    )

    assert first == shifted


def test_diagnostic_counts_preserve_duplicate_debt() -> None:
    diagnostics = [
        _diag("F401", "unused import", 1),
        _diag("F401", "unused import", 2),
    ]

    counts = diagnostic_counts("import x\nimport x\n", diagnostics)

    assert sum(counts.values()) == 2


def test_introduced_diagnostics_allow_exact_legacy_debt(monkeypatch) -> None:
    base = "legacy_line\n"
    current = "inserted\nlegacy_line\n"
    calls = iter(
        [
            [_diag("E501", "Line too long", 2)],
            [_diag("E501", "Line too long", 1)],
        ]
    )
    monkeypatch.setattr(_MODULE, "_ruff_diagnostics", lambda *_a: next(calls))

    introduced = _MODULE.introduced_diagnostics(base, current, path=Path("x.py"))

    assert not introduced


def test_introduced_diagnostics_fail_on_new_code_message_source(monkeypatch) -> None:
    base = "legacy_line\n"
    current = "legacy_line\nnew_bad_line\n"
    calls = iter(
        [
            [
                _diag("E501", "Line too long", 1),
                _diag("F841", "unused variable", 2),
            ],
            [_diag("E501", "Line too long", 1)],
        ]
    )
    monkeypatch.setattr(_MODULE, "_ruff_diagnostics", lambda *_a: next(calls))

    introduced = _MODULE.introduced_diagnostics(base, current, path=Path("x.py"))

    assert sum(introduced.values()) == 1
    assert ("F841", "unused variable", "new_bad_line") in introduced
