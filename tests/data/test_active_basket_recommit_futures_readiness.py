from __future__ import annotations

import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class _FakeWebSocket:
    def __init__(self) -> None:
        self.tokens: list[int] = []

    def set_tokens(self, tokens):
        self.tokens = list(tokens)
        return True

    def is_connected(self) -> bool:
        return True


def _basket() -> dict[str, object]:
    fut = "NFO:NIFTY26AUGFUT"
    ce = "NFO:NIFTY26AUG24250CE"
    pe = "NFO:NIFTY26AUG24250PE"
    return {
        "spot_symbol": "NSE:NIFTY",
        "spot_token": 256265,
        "futures_symbol": fut,
        "futures_token": 14866434,
        "selected_ce": ce,
        "selected_pe": pe,
        "selected_ce_token": 11545346,
        "selected_pe_token": 11545602,
        "option_symbols": [ce, pe],
        "option_tokens": [11545346, 11545602],
        "all_symbols": ["NSE:NIFTY", fut, ce, pe],
        "all_tokens": [256265, 14866434, 11545346, 11545602],
        "token_by_symbol": {
            "NSE:NIFTY": 256265,
            fut: 14866434,
            ce: 11545346,
            pe: 11545602,
        },
        "symbol_by_token": {
            256265: "NSE:NIFTY",
            14866434: fut,
            11545346: ce,
            11545602: pe,
        },
        "atm_strike": 24250,
    }


def test_identical_basket_recommit_preserves_active_future_live_readiness() -> None:
    """Periodic SSOT re-commit must not invalidate a healthy active-futures tick."""
    ws = _FakeWebSocket()
    mdm = MarketDataManager(kite=None, websocket=ws)
    basket = _basket()
    fut = str(basket["futures_symbol"])
    token = int(basket["futures_token"])

    mdm.set_active_contract_basket(basket)
    mdm._confirmed_subscriptions.add(token)  # noqa: SLF001
    mdm._emit_tick(  # noqa: SLF001
        fut,
        {
            "symbol": fut,
            "instrument_token": token,
            "ltp": 24_350.0,
            "bid": 24_349.5,
            "ask": 24_350.5,
            "depth_available": True,
            "timestamp": time.time(),
        },
        source="ws",
    )

    before = mdm.classify_live_tick_readiness(fut, token, max_age_s=60.0)
    generation = mdm._symbol_subscription_generation[fut]  # noqa: SLF001
    first_tick_generation = mdm._symbol_first_tick_generation[fut]  # noqa: SLF001
    cached = dict(mdm.get_latest_tick(fut) or {})
    assert before["ready"] is True
    assert cached.get("ltp") == 24_350.0

    mdm.set_active_contract_basket(basket)

    after = mdm.classify_live_tick_readiness(fut, token, max_age_s=60.0)
    assert mdm._symbol_subscription_generation[fut] == generation  # noqa: SLF001
    assert mdm._symbol_first_tick_generation[fut] == first_tick_generation  # noqa: SLF001
    assert after["ready"] is True
    assert (mdm.get_latest_tick(fut) or {}).get("ltp") == 24_350.0
    assert token in mdm._desired_tokens  # noqa: SLF001
