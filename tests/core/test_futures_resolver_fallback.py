"""Futures-context resolution must fall back to the InstrumentManager dump when
the committed basket has no futures_symbol yet (startup race). Async so it runs
under the repo conftest hook.
"""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.app import _resolve_active_futures_for_basket


class _IM:
    def __init__(self, loaded=True, contracts=None):
        self._loaded = loaded
        self._contracts = contracts if contracts is not None else [
            {"tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-30", "instrument_token": 15956226},
            {"tradingsymbol": "NIFTY26JULFUT", "expiry": "2026-07-28", "instrument_token": 15956300},
        ]

    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def get_future_contracts(self, underlying):
        assert underlying == "NIFTY"
        return list(self._contracts)


def _ctx(*, mdm_future=None, instrument_manager=None):
    mdm = SimpleNamespace(
        get_active_nifty_future_symbol_cached=lambda: mdm_future,
        resolve_active_nifty_future_symbol=lambda: mdm_future,
    )
    return SimpleNamespace(
        market_data_manager=mdm,
        active_trading_universe={},
        strategy_runner=None,
        strategy_manager=None,
        instrument_manager=instrument_manager,
    )


async def test_resolves_from_instrument_dump_when_basket_empty() -> None:
    # MDM/basket return nothing (startup), but the dump has futures -> resolve nearest.
    ctx = _ctx(mdm_future=None, instrument_manager=_IM())
    result = _resolve_active_futures_for_basket(ctx, None)
    assert result == "NFO:NIFTY26JUNFUT"


async def test_prefers_committed_basket_over_dump() -> None:
    # When MDM already has the future, that wins (no dump fallback needed).
    ctx = _ctx(mdm_future="NFO:NIFTY26JUNFUT", instrument_manager=_IM(contracts=[]))
    result = _resolve_active_futures_for_basket(ctx, None)
    assert result == "NFO:NIFTY26JUNFUT"


async def test_loads_instrument_manager_if_not_loaded() -> None:
    im = _IM(loaded=False)
    ctx = _ctx(mdm_future=None, instrument_manager=im)
    result = _resolve_active_futures_for_basket(ctx, None)
    assert result == "NFO:NIFTY26JUNFUT"
    assert im.is_loaded() is True


async def test_returns_empty_when_no_future_anywhere() -> None:
    # No MDM future and an empty dump -> graceful empty (no crash).
    ctx = _ctx(mdm_future=None, instrument_manager=_IM(contracts=[]))
    result = _resolve_active_futures_for_basket(ctx, None)
    assert result == ""


async def test_dump_fallback_failure_is_contained() -> None:
    class _Boom:
        def is_loaded(self):
            return True
        def get_future_contracts(self, _u):
            raise RuntimeError("dump unavailable")
    ctx = _ctx(mdm_future=None, instrument_manager=_Boom())
    # must not raise
    result = _resolve_active_futures_for_basket(ctx, None)
    assert result == ""
