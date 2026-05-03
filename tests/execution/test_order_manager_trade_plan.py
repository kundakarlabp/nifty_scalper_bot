from nifty_scalper_bot.execution.order_manager import OrderPreflightResult, TradePlan

def test_trade_plan_dataclass_defaults():
    plan = TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert plan.allow_market_entry is False
    assert isinstance(OrderPreflightResult(True), OrderPreflightResult)
