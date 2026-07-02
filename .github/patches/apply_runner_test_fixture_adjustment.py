from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "tests/strategies/test_runner_live_path_guards.py"
text = path.read_text(encoding="utf-8")

fixture_old = '''    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
'''
fixture_new = '''    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
    runner._signal_reject_cooldown_ts = {}
    runner._execution_reject_cooldown_ts = {}
    runner._order_failure_cooldown_until = {}
    runner._order_failure_cooldown_seconds = 30.0
    runner._last_regime_inputs_by_symbol = {}
    runner._lock = __import__("threading").RLock()
    runner._execution_state_lock = runner._lock
    runner._execution_state_by_symbol = {}
'''
if text.count(fixture_old) != 1:
    raise RuntimeError("runner fixture anchor mismatch")
text = text.replace(fixture_old, fixture_new, 1)

test_old = '''def test_runner_normalizes_one_lot_to_exchange_quantity() -> None:
    runner = _build_runner()
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.9,
        reason='test',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={},
    )
    runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='t1',
    )
    assert runner._order_manager.last_quantity == 65
'''
test_new = '''def test_runner_normalizes_one_lot_to_exchange_quantity() -> None:
    runner = _build_runner()
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26APR23800PE',
        quantity=1,
        confidence=0.9,
        reason='SMC',
        stop_loss=100.0,
        take_profit=120.0,
        metadata={
            'direction_score': 8.0,
            'strategy_score': 8.0,
            'option_score': 8.0,
            'data_score': 8.0,
            'rr_score': 8.0,
        },
    )
    result = runner._handle_entry_signal_inner(
        signal,
        base_symbol='NFO:NIFTY26APR23800PE',
        trade_symbol='NFO:NIFTY26APR23800PE',
        trade_price=110.0,
        timestamp=datetime.now(timezone.utc),
        trace_id='t1',
    )
    assert result.accepted is True, result
    assert runner._order_manager.last_quantity == 65
'''
if text.count(test_old) != 1:
    raise RuntimeError("quantity regression anchor mismatch")
path.write_text(text.replace(test_old, test_new, 1), encoding="utf-8")
Path(__file__).unlink()
