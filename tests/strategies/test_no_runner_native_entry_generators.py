from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.strategies import runner


def test_runner_has_no_native_entry_generators_outside_strategy_manager() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "RUNNER_ENABLE_PREMIUM_SQUEEZE" not in source
    assert "RUNNER_ENABLE_LEGACY_VWAP_CROSSOVER" not in source
    assert "def _maybe_generate_premium_squeeze_signal(" not in source
    assert "_premium_squeeze_last_signal_ts" not in source
    assert "premium_momentum_squeeze" not in source
    assert "vwap_crossover_up" not in source
    assert "vwap_crossover_down" not in source
    assert "_force_signal_enabled" not in source
