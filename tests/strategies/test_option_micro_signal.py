from __future__ import annotations

from nifty_scalper_bot.strategies.option_micro_signal import OptionMicroSignal


def make_signal() -> OptionMicroSignal:
    return OptionMicroSignal(
        spread_limit=2.0,
        min_depth=10,
        snmom_threshold=0.5,
        microvol_threshold=0.01,
        adverse_ticks=2,
    )


def test_long_entry_on_positive_momentum() -> None:
    signal = make_signal()
    signal.on_tick(
        bid=100.0,
        ask=101.0,
        last_price=101.0,
        last_size=5,
        depth=20,
        ts_ns=0,
    )
    decision = signal.on_tick(
        bid=101.0,
        ask=102.0,
        last_price=102.0,
        last_size=4,
        depth=20,
        ts_ns=200_000_000,
    )
    assert decision.enter is True
    assert decision.side == "LONG"


def test_no_entry_when_spread_wide() -> None:
    signal = make_signal()
    signal.on_tick(
        bid=100.0,
        ask=101.0,
        last_price=101.0,
        last_size=5,
        depth=20,
        ts_ns=0,
    )
    decision = signal.on_tick(
        bid=98.0,
        ask=102.5,
        last_price=102.5,
        last_size=3,
        depth=20,
        ts_ns=200_000_000,
    )
    assert not decision.enter


def test_exit_on_adverse_move() -> None:
    signal = make_signal()
    signal.on_tick(
        bid=100.0,
        ask=101.0,
        last_price=101.0,
        last_size=5,
        depth=20,
        ts_ns=0,
    )
    entry = signal.on_tick(
        bid=101.0,
        ask=102.0,
        last_price=102.0,
        last_size=4,
        depth=20,
        ts_ns=200_000_000,
    )
    assert entry.enter is True
    exit_decision = signal.on_tick(
        bid=99.0,
        ask=100.0,
        last_price=99.0,
        last_size=2,
        depth=20,
        ts_ns=400_000_000,
        position_side="LONG",
        entry_mid=101.5,
    )
    assert exit_decision.exit is True
    assert exit_decision.reason == "HARD_STOP"


def test_invalid_quote_returns_hold() -> None:
    signal = make_signal()
    decision = signal.on_tick(
        bid=102.0,
        ask=101.0,
        last_price=101.0,
        last_size=1,
        depth=10,
        ts_ns=0,
    )
    assert decision.enter is False
    assert decision.exit is False


class _PortfolioStub:
    def __init__(self, open_position: bool) -> None:
        self._open_position = open_position

    def has_open_position(self, symbol: str) -> bool:
        return self._open_position


def test_entry_blocked_by_portfolio_position() -> None:
    signal = make_signal()
    signal.on_tick(
        bid=100.0,
        ask=101.0,
        last_price=101.0,
        last_size=5,
        depth=20,
        ts_ns=0,
    )
    decision = signal.on_tick(
        bid=101.0,
        ask=102.0,
        last_price=102.0,
        last_size=4,
        depth=20,
        ts_ns=200_000_000,
        symbol="NFO:NIFTY26FEB22500CE",
        portfolio=_PortfolioStub(open_position=True),
    )
    assert decision.enter is False


def test_stateless_exit_without_internal_position() -> None:
    signal = make_signal()
    signal.on_tick(
        bid=100.0,
        ask=101.0,
        last_price=101.0,
        last_size=5,
        depth=20,
        ts_ns=0,
    )
    decision = signal.on_tick(
        bid=99.0,
        ask=100.0,
        last_price=99.0,
        last_size=2,
        depth=20,
        ts_ns=200_000_000,
        position_side="LONG",
        entry_mid=101.0,
    )
    assert decision.exit is True
