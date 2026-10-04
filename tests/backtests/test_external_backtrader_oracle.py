"""Independent Backtrader oracle must agree on resolved long-trade gross P&L."""

import pytest
from scripts.external_backtrader_oracle import backtrader_gross_pnl


def test_backtrader_oracle_replays_long_round_trip_gross_pnl():
    trades = [
        {
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 75,
            "gross_pnl": 750.0,
        },
        {
            "entry_price": 120.0,
            "exit_price": 115.0,
            "quantity": 50,
            "gross_pnl": -250.0,
        },
    ]
    assert backtrader_gross_pnl(trades) == pytest.approx(500.0)


def test_backtrader_oracle_rejects_invalid_trade():
    with pytest.raises(ValueError, match="external_oracle_trade_invalid"):
        backtrader_gross_pnl(
            [{"entry_price": 0, "exit_price": 10, "quantity": 1}]
        )
