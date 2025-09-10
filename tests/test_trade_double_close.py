import logging
from src.risk.session import Trade

def test_trade_double_close(caplog):
    trade = Trade("SYM", "BUY", 100.0, 1, "oid1", atr_at_entry=5.0)
    trade.close(105.0)
    with caplog.at_level(logging.WARNING):
        trade.close(106.0)
    # second close shouldn't change exit_price
    assert trade.exit_price == 105.0
    assert "already closed" in caplog.text
