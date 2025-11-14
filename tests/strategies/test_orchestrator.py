from __future__ import annotations

from dataclasses import dataclass

from nifty_scalper_bot.strategies.orchestrator import StrategyOrchestrator
from nifty_scalper_bot.strategies.signal_generator import Signal


@dataclass
class _DummyPosition:
    symbol: str


class _OrderManagerStub:
    def __init__(self) -> None:
        self.reason: str | None = None

    def set_last_skip_reason(self, reason: str) -> None:
        self.reason = reason

    def consume_skip_reason(self) -> str | None:
        return self.reason


class _PositionManagerStub:
    def __init__(self, exposure: float = 0.0, symbols: tuple[str, ...] = ()) -> None:
        self._exposure = exposure
        self._symbols = symbols

    def get_total_exposure(self) -> float:
        return self._exposure

    def get_all_positions(self) -> list[_DummyPosition]:
        return [_DummyPosition(symbol=s) for s in self._symbols]


class _RiskStub:
    def __init__(self, balance: float) -> None:
        self.current_balance = balance


def _make_signal(strategy: str, action: str = "BUY") -> Signal:
    return Signal(
        action=action,
        symbol="NIFTY",
        quantity=1,
        confidence=0.5,
        reason="test",
        stop_loss=10.0,
        take_profit=20.0,
        metadata={"strategy": strategy},
    )


def test_orchestrator_blocks_when_capital_exceeded() -> None:
    risk = _RiskStub(balance=100_000.0)
    order_manager = _OrderManagerStub()
    orchestrator = StrategyOrchestrator(
        risk_manager=risk, order_manager=order_manager, data_hub=None
    )
    orchestrator.register_strategy(
        "capital", capital_fraction=0.1, correlation_tags=("x",)
    )
    signal = _make_signal("capital")
    pos_manager = _PositionManagerStub(exposure=20_000.0)

    result = orchestrator.filter_signal(signal, {}, pos_manager)
    assert result is None
    assert order_manager.reason == "orchestrator_capital"


def test_orchestrator_blocks_correlated_underlying() -> None:
    risk = _RiskStub(balance=100_000.0)
    orchestrator = StrategyOrchestrator(risk_manager=risk)
    orchestrator.register_strategy(
        "momentum", capital_fraction=0.5, correlation_tags=("m",)
    )
    signal = _make_signal("momentum")
    pos_manager = _PositionManagerStub()

    orchestrator.notify_submission(signal, "NIFTY")
    blocked = orchestrator.filter_signal(signal, {}, pos_manager)
    assert blocked is None

    orchestrator.notify_exit("NIFTY")
    allowed = orchestrator.filter_signal(
        signal, {"futures_volume_ratio": 1.0}, pos_manager
    )
    assert allowed is signal


def test_orchestrator_requires_futures_context_when_datahub() -> None:
    risk = _RiskStub(balance=50_000.0)
    order_manager = _OrderManagerStub()
    orchestrator = StrategyOrchestrator(
        risk_manager=risk, order_manager=order_manager, data_hub=object()
    )
    orchestrator.register_strategy("futures", capital_fraction=0.2, correlation_tags=())
    signal = _make_signal("futures")
    pos_manager = _PositionManagerStub(exposure=0.0)

    result = orchestrator.filter_signal(signal, {}, pos_manager)
    assert result is None
    assert order_manager.reason == "orchestrator_futures"
