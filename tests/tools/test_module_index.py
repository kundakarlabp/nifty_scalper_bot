"""Regression tests for the source-derived critical-module index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import module_index


def _fixture_repo(root: Path) -> Path:
    manifest = root / "docs/architecture/agent_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "runtime_path": ["src/nifty_scalper_bot/core/zeta.py"],
                "module_documentation": ["dashboard/alpha.py"],
                "owners": [],
            }
        ),
        encoding="utf-8",
    )
    for path, doc in (
        (
            "src/nifty_scalper_bot/core/zeta.py",
            "Purpose: Parse prepared ticks.\n\nResponsibilities: Preserve source.\n"
            "Safety: Never place broker orders.",
        ),
        (
            "dashboard/alpha.py",
            "File purpose:\n    Show ready | blocked dashboard state.\n"
            "Boundaries: Read only.",
        ),
    ):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            f'"""{doc}"""\n\nraise RuntimeError("must never import")\n',
            encoding="utf-8",
        )
    return root


def test_generated_index_is_deterministic_and_uses_manifest(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    first = module_index.render(root)
    assert first == module_index.render(root)
    assert "**Documented critical modules:** 2" in first
    assert first.index("dashboard/alpha.py") < first.index("core/zeta.py")
    assert r"Show ready \| blocked dashboard state." in first
    assert "Parse prepared ticks." in first


def test_stale_or_missing_index_is_rejected(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    assert module_index.main(["--root", str(root), "--check"]) == 1
    assert module_index.main(["--root", str(root), "--write"]) == 0
    assert module_index.main(["--root", str(root), "--check"]) == 0
    source = root / "src/nifty_scalper_bot/core/zeta.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "Parse prepared ticks.", "Parse selected option ticks."
        ),
        encoding="utf-8",
    )
    assert module_index.main(["--root", str(root), "--check"]) == 1


def test_invalid_manifest_module_fails_closed(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    (root / "dashboard/alpha.py").unlink()
    with pytest.raises(FileNotFoundError):
        module_index.render(root)


def test_committed_index_matches_headers() -> None:
    root = Path(__file__).resolve().parents[2]
    assert module_index.is_current(root), (
        "Regenerate docs/MODULE_INDEX.md using "
        "python -m scripts.module_index --write"
    )
