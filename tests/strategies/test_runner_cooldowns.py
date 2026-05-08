from __future__ import annotations

from pathlib import Path


def test_on_tick_error_phase_updates_for_premium_squeeze() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'phase = "phase8_premium_squeeze"' in source
    assert 'phase = "phase8_vwap_crossover"' in source
    assert 'phase = "phase9_strategy_manager"' in source
    assert 'phase = "phase10_signal_execution"' in source
    assert 'RUNNER_ON_TICK_ERROR symbol=%s phase=%s error_type=%s error=%s' in source
    assert 'exc_info=True' in source


def test_cooldown_rejection_diagnostics_logged() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'COOLDOWN_REJECTED reason=reason_cooldown' in source
    assert 'COOLDOWN_REJECTED reason=underlying_cooldown' in source



def test_premium_squeeze_generation_does_not_stamp_reason_cooldown() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    marker = 'def _maybe_generate_premium_squeeze_signal('
    start = source.index(marker)
    end = source.index('\n\n    def _handle_signal(', start)
    premium_block = source[start:end]
    assert '_reason_last_signal_ts' not in premium_block
    assert '_underlying_last_signal_ts' not in premium_block
    assert '_premium_squeeze_last_signal_ts' not in premium_block


def test_reason_cooldown_stamped_only_after_order_id() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    order_if = source.index('if order_id:')
    stamped = source.index('self._reason_last_signal_ts[underlying_reason_key] = now_epoch')
    assert stamped > order_if


def test_min_depth_float_env_parses() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'min_depth_qty = int(float(os.getenv("ORDER_MIN_DEPTH_QTY", os.getenv("MIN_DEPTH_QTY", "0")) or 0))' in source
