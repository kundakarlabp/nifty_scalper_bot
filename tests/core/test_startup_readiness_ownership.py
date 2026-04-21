from __future__ import annotations

from pathlib import Path


def test_app_startup_does_not_mark_runner_ready_directly() -> None:
    source = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    assert "runner.mark_ready(" not in source


def test_app_startup_does_not_patch_indicators_ready_before_mdm_gate() -> None:
    source = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    assert "indicators_ready = bool(ready_symbols)" not in source
