from __future__ import annotations

from src.config import settings
from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


class DummyLeg:
    def __init__(self, symbol: str, price: float, qty: int) -> None:
        self.symbol = symbol
        self.limit_price = price
        self.avg_price = None
        self.qty = qty


class DummyFsm:
    def __init__(self, legs) -> None:
        self._legs = legs

    def open_legs(self):
        return self._legs


class DummyOE:
    def __init__(self, fsms) -> None:
        self._fsms = fsms

    def open_trades(self):
        return self._fsms


def test_notional_premium(monkeypatch) -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    legs = [DummyLeg("A", 10.0, 5), DummyLeg("B", 20.0, 3)]
    runner.order_executor = DummyOE([DummyFsm(legs)])
    monkeypatch.setattr(settings, "exposure_basis", "premium", raising=False)
    assert runner._notional_rupees() == 10.0 * 5 + 20.0 * 3


def test_notional_underlying(monkeypatch) -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    legs = [DummyLeg("SYM", 10.0, 75), DummyLeg("SYM", 10.0, 75)]
    runner.order_executor = DummyOE([DummyFsm(legs)])
    runner.last_spot = 100.0
    monkeypatch.setattr(settings, "exposure_basis", "underlying", raising=False)
    monkeypatch.setattr(settings.instruments, "nifty_lot_size", 75, raising=False)
    assert runner._notional_rupees() == 100.0 * 75 * 2
