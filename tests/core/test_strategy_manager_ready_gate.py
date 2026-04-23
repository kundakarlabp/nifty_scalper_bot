from __future__ import annotations

from pathlib import Path


def test_strategy_manager_blocks_when_indicators_not_ready() -> None:
    source = Path('src/nifty_scalper_bot/core/strategy_manager.py').read_text(
        encoding='utf-8'
    )

    assert 'signal_blocked_indicators_not_ready' in source
    assert 'getattr(self._data_hub, "indicators_ready", False)' in source


def test_app_sets_data_hub_indicators_from_hard_ready_gate() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')

    assert 'indicators_ready_for_trading = hard_ready' in source
    assert 'ctx.data_hub.indicators_ready = (' in source
