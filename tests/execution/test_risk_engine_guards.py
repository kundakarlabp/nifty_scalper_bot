from dataclasses import dataclass

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.risk_manager import RiskManager


@dataclass
class _Position:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_loss: float | None
    take_profit: float | None


class _PM:
    def __init__(self) -> None:
        self._positions: list[_Position] = []

    def get_open_positions(self):
        return list(self._positions)

    def get_position(self, symbol: str):
        for p in self._positions:
            if p.symbol == symbol:
                return p
        return None


def test_risk_per_trade_guard_rejects_large_loss() -> None:
    pm = _PM()
    cfg = RiskConfig()
    rm = RiskManager(cfg, pm, 100_000.0)
    ok, reason = rm.validate_new_position(
        symbol='NFO:NIFTY26JAN25000CE',
        side='LONG',
        quantity=100,
        entry_price=200.0,
        stop_loss=150.0,
        take_profit=300.0,
    )
    assert ok is False
    assert '1%' in reason
