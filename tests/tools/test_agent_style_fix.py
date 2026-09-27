from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_style_module():
    path = ROOT / "scripts" / "agent_style_fix.py"
    spec = importlib.util.spec_from_file_location("agent_style_fix_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Result:
    returncode = 0
    stdout = ""
    stderr = ""


def test_style_fix_filters_to_repository_python_surfaces() -> None:
    module = _load_style_module()

    files = module.changed_python_files(
        ROOT,
        base_ref="origin/main",
        explicit=(
            "src/nifty_scalper_bot/a.py",
            "tests/test_a.py",
            "docs/readme.md",
            "vendor/tool.py",
        ),
    )

    assert files == (
        "src/nifty_scalper_bot/a.py",
        "tests/test_a.py",
    )


def test_style_fix_auto_formats_new_file(monkeypatch) -> None:
    module = _load_style_module()
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(module, "_base_content", lambda *_args, **_kwargs: None)

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        return _Result()

    monkeypatch.setattr(module, "_run", fake_run)

    actions = module._fix_file(
        ROOT,
        base_ref="origin/main",
        path="tests/new_test.py",
    )

    assert actions == ("ruff", "black")
    assert any("ruff" in call for call in calls)
    assert any("black" in call for call in calls)


def test_style_fix_does_not_rewrite_legacy_debt(monkeypatch) -> None:
    module = _load_style_module()
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        module,
        "_base_content",
        lambda *_args, **_kwargs: "legacy = True\n",
    )
    monkeypatch.setattr(
        module,
        "_snapshot_clean",
        lambda *_args, **_kwargs: False,
    )

    def fake_run(argv, **_kwargs):
        calls.append(tuple(argv))
        return _Result()

    monkeypatch.setattr(module, "_run", fake_run)

    actions = module._fix_file(
        ROOT,
        base_ref="origin/main",
        path="src/nifty_scalper_bot/legacy.py",
    )

    assert actions == (
        "ruff-skipped-legacy-debt",
        "black-skipped-legacy-debt",
    )
    assert calls == []
