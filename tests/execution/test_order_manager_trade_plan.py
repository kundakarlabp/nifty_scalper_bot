from nifty_scalper_bot.execution.order_manager import OrderPreflightResult, TradePlan

def test_trade_plan_dataclass_defaults():
    plan = TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert plan.allow_market_entry is False
    assert isinstance(OrderPreflightResult(True), OrderPreflightResult)


def test_submit_trade_plan_passes_allow_market_entry() -> None:
    manager = type('M', (), {})()
    captured = {}
    def place_managed_order(**kwargs):
        captured.update(kwargs)
        return 'oid-1'
    manager.place_managed_order = place_managed_order
    manager._preflight_trade_plan = lambda _plan: OrderPreflightResult(True)
    from nifty_scalper_bot.execution.order_manager import OrderManager
    plan = TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0, allow_market_entry=True)
    result = OrderManager.submit_trade_plan(manager, plan)  # type: ignore[arg-type]
    assert result == 'oid-1'
    assert captured['allow_market_entry'] is True
