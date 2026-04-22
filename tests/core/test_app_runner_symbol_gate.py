from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.app import (
    _gate_runner_symbol_add,
    _history_lookback_minutes,
    _symbol_history_requirement,
)


class _Runner:
    def __init__(self, required: int = 50) -> None:
        self._required_candles = required
        self.added: list[str] = []

    def add_symbol(self, sym: str) -> None:
        self.added.append(sym)


class _Mdm:
    def __init__(self, bars_by_symbol: dict[str, int], required: int = 50) -> None:
        self._min_required_bars = required
        self._bars_by_symbol = bars_by_symbol

    def get_ohlc_bars(self, sym: str):
        return [object()] * int(self._bars_by_symbol.get(sym, 0))


def test_startup_runner_add_gate_resolved_and_unresolved() -> None:
    runner = _Runner(required=55)
    mdm = _Mdm({"NSE:NIFTY": 80, "NFO:NIFTY24500CE": 20}, required=60)
    ctx = SimpleNamespace(strategy_runner=runner, market_data_manager=mdm)
    pending: set[str] = set()

    token_map = {"NSE:NIFTY": 256265, "NFO:NIFTY24500CE": 1234}
    unresolved: list[str] = []
    for sym in ["NSE:NIFTY", "NFO:NIFTY24500CE", "NFO:NIFTY24500PE"]:
        tok = token_map.get(sym)
        if not tok:
            unresolved.append(sym)
            continue
        _gate_runner_symbol_add(ctx, sym, pending)

    assert runner.added == ["NSE:NIFTY"]
    assert "NFO:NIFTY24500CE" in pending
    assert unresolved == ["NFO:NIFTY24500PE"]


def test_active_basket_deferred_add_then_single_promotion() -> None:
    runner = _Runner(required=50)
    mdm = _Mdm({"NFO:NIFTY24600CE": 30}, required=50)
    ctx = SimpleNamespace(strategy_runner=runner, market_data_manager=mdm)
    pending: set[str] = set()

    assert _gate_runner_symbol_add(ctx, "NFO:NIFTY24600CE", pending) is False
    assert runner.added == []
    assert "NFO:NIFTY24600CE" in pending

    mdm._bars_by_symbol["NFO:NIFTY24600CE"] = 75
    assert _gate_runner_symbol_add(ctx, "NFO:NIFTY24600CE", pending) is True
    assert runner.added == ["NFO:NIFTY24600CE"]
    assert "NFO:NIFTY24600CE" not in pending


def test_symbol_history_requirement_uses_max_threshold() -> None:
    runner = _Runner(required=75)
    mdm = _Mdm({}, required=60)
    ctx = SimpleNamespace(strategy_runner=runner, market_data_manager=mdm)
    assert _symbol_history_requirement(ctx) == 75


def test_history_fetch_lookback_grows_with_requirement() -> None:
    assert _history_lookback_minutes(50) == 180
    assert _history_lookback_minutes(240) > 240
    assert _history_lookback_minutes(500) > _history_lookback_minutes(240)
