from pathlib import Path


def test_strategy_manager_enter_is_debug() -> None:
    source = Path('src/nifty_scalper_bot/core/strategy_manager.py').read_text(
        encoding='utf-8'
    )
    assert 'log.debug(\n            "STRATEGY_MANAGER_ENTER"' in source


def test_orchestrator_signal_blocked_not_warning() -> None:
    source = Path('src/nifty_scalper_bot/strategies/orchestrator.py').read_text(
        encoding='utf-8'
    )
    assert 'self._logger.debug(\n                "EVENT|signal_blocked|symbol=%s|reason=%s"' in source
