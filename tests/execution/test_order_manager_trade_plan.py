from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager import OrderManager, OrderPreflightResult, TradePlan


def _manager_stub():
    m = SimpleNamespace()
    m._logger = SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None)
    return m


def test_trade_plan_dataclass_defaults():
    plan = TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert plan.allow_market_entry is False
    assert isinstance(OrderPreflightResult(True), OrderPreflightResult)


def test_submit_trade_plan_protected_price_invalidates_buy_bracket() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(True)
    m._protected_limit_price = lambda p: 111.0
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=95.0, take_profit=105.0))
    assert out.accepted is False
    assert out.reason == 'protected_price_invalidates_bracket'
    assert out.broker_attempted is False
    assert out.details['violation'] == 'take_profit_below_or_equal_entry'


def test_submit_trade_plan_protected_price_invalidates_sell_bracket() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(True)
    m._protected_limit_price = lambda p: 99.0
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='SELL', quantity=75, entry_price=100.0, stop_loss=101.0, take_profit=98.0))
    assert out.accepted is False
    assert out.reason == 'protected_price_invalidates_bracket'
    assert out.broker_attempted is False


def test_preflight_reject_does_not_attempt_broker() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(False, 'quote_unavailable', {})
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0))
    assert out.reason == 'quote_unavailable'
    assert out.broker_attempted is False


def test_managed_order_local_reject_does_not_attempt_broker() -> None:
    m = _manager_stub()
    m._lot_size_for_symbol = lambda s: 50
    out = OrderManager.place_managed_order_result(m, symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert out.reason == 'invalid_lot_quantity'
    assert out.broker_attempted is False


def test_managed_order_uses_last_decision_for_broker_attempted_none() -> None:
    m = _manager_stub()
    m._lot_size_for_symbol = lambda s: 75

    def _place_order(**kwargs):
        m._last_order_decision = {'block_reason': 'risk_manager_blocked', 'details': {'x': 1}, 'broker_attempted': True}
        return None

    m.place_order = _place_order
    out = OrderManager.place_managed_order_result(m, symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert out.reason == 'risk_manager_blocked'
    assert out.broker_attempted is True


def test_stale_last_order_decision_does_not_leak() -> None:
    m = _manager_stub()
    m._lot_size_for_symbol = lambda s: 75
    m._last_order_decision = {'block_reason': 'stale_old_reason', 'broker_attempted': True}
    m.place_order = lambda **kwargs: None
    out = OrderManager.place_managed_order_result(m, symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0, trace_id='t1')
    assert out.reason == 'place_order_rejected_without_decision'
    assert out.broker_attempted is False
