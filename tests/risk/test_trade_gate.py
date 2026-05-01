from nifty_scalper_bot.core.signal_arbitrator import TradeDecision
from nifty_scalper_bot.risk.trade_gate import TradeGate


def decision():
    return TradeDecision('BUY','CE','SYM','NIFTY',8,0.8,100,92,113,1.6,['ok'],[],{},'t',1.0)


def test_pause_blocks():
    assert not TradeGate().evaluate(decision(), {'paused': True}).allowed


def test_daily_loss_blocks():
    assert not TradeGate().evaluate(decision(), {'daily_loss': -1000, 'max_daily_loss': 500}).allowed


def test_duplicate_blocks():
    assert not TradeGate().evaluate(decision(), {'open_symbols': ['SYM']}).allowed


def test_valid_passes():
    assert TradeGate().evaluate(decision(), {'market_open': True, 'live_orders_armed': True}).allowed
