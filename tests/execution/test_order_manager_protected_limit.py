from nifty_scalper_bot.execution.order_manager import TradePlan

def test_trade_plan_side_values():
    assert TradePlan(symbol='X', side='BUY', quantity=1, entry_price=1, stop_loss=1, take_profit=1).side == 'BUY'
