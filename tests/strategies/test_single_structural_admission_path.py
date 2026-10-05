from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.core import strategy_manager


def test_strategy_manager_has_single_post_combine_admission_path() -> None:
    source = Path(strategy_manager.__file__).read_text(encoding="utf-8")

    assert "self._filter_signal(combined)" not in source
    assert "orchestrator.filter_signal(" not in source
    assert 'metadata.get("kelly_fraction"' not in source
