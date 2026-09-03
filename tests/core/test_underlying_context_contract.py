from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_context_builder import build_strategy_history_context
from nifty_scalper_bot.core.underlying_context_contract import (
    configured_ohlc_capacity,
    ensure_mdm_ohlc_capacity,
    resolve_active_underlying_symbols,
)


FUTURE = "NFO:NIFTY26SEPFUT"
SPOT = "NSE:NIFTY"
OPTION = "NFO:NIFTY2690823950CE"


class _Hub:
    def get_active_contract_basket(self):
        return {
            "spot_symbol": SPOT,
            "futures_symbol": FUTURE,
            "selected_ce": OPTION,
        }

    def get_ohlc_bars(self, _symbol: str):
        return [{"timestamp": index, "close": 100.0 + index} for index in range(35)]


class _Indicator:
    def get_history(self, _symbol: str):
        return []


async def test_option_history_context_carries_active_underlying_symbols() -> None:
    context = build_strategy_history_context(
        symbol=OPTION,
        indicator_engine=_Indicator(),
        data_hub=_Hub(),
        runner_context={},
    )

    assert context["futures_symbol"] == FUTURE
    assert context["spot_symbol"] == SPOT


async def test_runner_context_wins_over_basket_symbol_identity() -> None:
    new_future = "NFO:NIFTY26OCTFUT"
    spot, future = resolve_active_underlying_symbols(
        _Hub(),
        {"spot_symbol": SPOT, "futures_symbol": new_future},
    )

    assert spot == SPOT
    assert future == new_future


async def test_completed_ohlc_capacity_is_independent_of_raw_tick_capacity(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    manager = SimpleNamespace(
        _cache_len=250,
        _ohlc_cache_len=250,
        _ohlc={FUTURE: deque(range(250), maxlen=250)},
        _engines={},
    )

    capacity = ensure_mdm_ohlc_capacity(manager)

    assert capacity == configured_ohlc_capacity() == 500
    assert manager._cache_len == 250
    assert manager._ohlc_cache_len == 500
    assert manager._ohlc[FUTURE].maxlen == 500
    assert len(manager._ohlc[FUTURE]) == 250
