from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_context_fast_path import (
    CONTEXT_REQUIRED_INDICATORS,
    _generate_context_only,
)


class _Indicators:
    def __init__(self) -> None:
        self.calls: list[tuple[str, set[str]]] = []

    def get_indicators(self, symbol: str, required: set[str]):
        self.calls.append((symbol, set(required)))
        return {
            "ltp": 24450.0,
            "close": 24450.0,
            "vwap": 24440.0,
            "ema_fast": 24448.0,
            "ema_slow": 24442.0,
            "volume": 1000.0,
            "avg_volume": 800.0,
        }


def _manager():
    engine = _Indicators()
    updates: list[dict] = []
    augmented: list[dict] = []
    manager = SimpleNamespace(
        _indicator_engine=engine,
        _last_no_signal_decision_by_symbol={"NSE:NIFTY": object()},
        _augment_futures_metrics=lambda indicators: augmented.append(dict(indicators)),
        _update_context_snapshot=lambda **kwargs: updates.append(kwargs),
    )
    return manager, engine, updates, augmented


def test_context_fast_path_uses_compact_indicator_contract() -> None:
    manager, engine, updates, augmented = _manager()

    assert _generate_context_only(manager, "NSE:NIFTY", 24451.0, "spot_context") is None

    assert len(engine.calls) == 1
    symbol, required = engine.calls[0]
    assert symbol == "NSE:NIFTY"
    assert required == set(CONTEXT_REQUIRED_INDICATORS)
    assert "bos_confirmed" not in required
    assert "liquidity_sweep_confirmed" not in required
    assert "selected_ce" not in required
    assert augmented
    assert updates and updates[0]["role"] == "spot_context"
    indicators = updates[0]["indicators"]
    assert indicators["ltp"] == 24450.0
    assert indicators["price"] == 24451.0
    assert indicators["symbol_role"] == "spot_context"
    assert "NSE:NIFTY" not in manager._last_no_signal_decision_by_symbol


def test_context_fast_path_accepts_mapping_like_indicator_result() -> None:
    manager, engine, updates, _ = _manager()

    class _MappingLike:
        def items(self):
            return {"ltp": 25000.0, "vwap": 24990.0}.items()

    engine.get_indicators = lambda _symbol, _required: _MappingLike()

    _generate_context_only(manager, "NFO:NIFTY26AUGFUT", 25001.0, "futures_context")

    assert updates[0]["indicators"]["ltp"] == 25000.0
    assert updates[0]["indicators"]["price"] == 25001.0
    assert updates[0]["role"] == "futures_context"


def test_context_fast_path_fails_closed_to_empty_indicator_mapping() -> None:
    manager, engine, updates, _ = _manager()
    engine.get_indicators = lambda _symbol, _required: None

    _generate_context_only(manager, "NSE:NIFTY", 24451.0, "spot_context")

    indicators = updates[0]["indicators"]
    assert indicators["ltp"] == 24451.0
    assert indicators["close"] == 24451.0
    assert indicators["price"] == 24451.0
