from datetime import datetime, timezone

from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


class _Broker:
    def __init__(self, mode='ok'):
        self.mode = mode
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        if self.mode == 'reject':
            raise RuntimeError('invalid payload 400')
        return {'order_id': 'OID1'}


class _Pos:
    def has_open_position(self, symbol):
        return False


class _Limiter:
    pass


def _om(mode='ok'):
    return OrderManager(_Broker(mode), _Pos(), _Limiter())


def test_duplicate_signal_paths_and_new_signal_not_blocked(monkeypatch):
    om = _om('ok')
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_confirm_fill_fast', lambda order_id, timeout_ms=300: False)

    assert om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='s1') == 'OID1'

    assert om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='s1') is None

    om._mark_signal_pending('s2')
    assert om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='s2') is None


def test_signal_lifecycle_failure_and_success(monkeypatch):
    om_fail = _om('reject')
    monkeypatch.setattr(om_fail, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om_fail, '_validate_live_execution_safety', lambda: True)
    assert om_fail.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='sf') is None
    assert 'sf' not in om_fail._seen_signal_ids
    assert 'sf' not in om_fail._pending_signal_ids

    om_ok = _om('ok')
    monkeypatch.setattr(om_ok, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om_ok, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om_ok, '_confirm_fill_fast', lambda order_id, timeout_ms=300: False)
    assert om_ok.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='ss') == 'OID1'
    assert 'ss' in om_ok._seen_signal_ids
    assert 'ss' not in om_ok._pending_signal_ids


def test_timestamp_seconds_variants():
    now = datetime.now(timezone.utc)
    assert OrderManager._timestamp_seconds(123.0) == 123.0
    assert OrderManager._timestamp_seconds(now) > 0
    assert OrderManager._timestamp_seconds('2026-05-19T10:00:00Z') > 0
    assert OrderManager._timestamp_seconds('bad-ts') == 0.0
