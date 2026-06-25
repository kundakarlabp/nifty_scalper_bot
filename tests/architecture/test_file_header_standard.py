from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_LABELS = (
    "File purpose:",
    "Key responsibilities:",
    "Operational constraints:",
)

PYTHON_FILES = (
    "src/nifty_scalper_bot/execution/order_manager.py",
    "src/nifty_scalper_bot/execution/bracket_manager.py",
    "src/nifty_scalper_bot/execution/adaptive_trailing.py",
    "src/nifty_scalper_bot/execution/ownership.py",
    "src/nifty_scalper_bot/execution/safe_order_manager.py",
    "src/nifty_scalper_bot/execution/lifecycle_manager.py",
    "src/nifty_scalper_bot/deployment_main.py",
)

COMMENT_HEADER_FILES = (
    "Dockerfile",
    "railway.toml",
    ".github/workflows/live-exit-safety-ci.yml",
)


def test_critical_python_files_have_descriptive_module_headers() -> None:
    failures: list[str] = []
    for relative in PYTHON_FILES:
        path = ROOT / relative
        module = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(module, clean=False) or ""
        missing = [label for label in REQUIRED_LABELS if label not in docstring]
        if missing:
            failures.append(f"{relative}: missing {', '.join(missing)}")
    assert not failures, failures


def test_critical_configuration_files_have_descriptive_comment_headers() -> None:
    failures: list[str] = []
    for relative in COMMENT_HEADER_FILES:
        path = ROOT / relative
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12])
        missing = [label for label in REQUIRED_LABELS if label not in header]
        if missing:
            failures.append(f"{relative}: missing {', '.join(missing)}")
    assert not failures, failures


def test_file_header_editing_rule_is_documented() -> None:
    standard = (ROOT / "docs" / "FILE_HEADER_STANDARD.md").read_text(
        encoding="utf-8"
    )
    for label in REQUIRED_LABELS:
        assert label in standard
    assert "When a listed canonical BO or deployment file is edited" in standard
