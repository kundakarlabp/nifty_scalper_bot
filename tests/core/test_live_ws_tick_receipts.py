from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.core.runtime_reliability_hardening import apply_patches
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    pass


def _seed_future_cache(mdm: MarketDataManager, symbol: str, token: int) -> None:
    future = datetime.now(timezone.utc).timestamp() + 60.0
    mdm._latest_ticks[mdm._canonical_symbol(symbol)] = {  # noqa: SLF001
        "symbol": symbol,
        "instrument_token": token,
        "ltp": 100.0,
        "timestamp": future,
        "exchange_timestamp": future,
        "source": "ws",
        "bid": 99.9,
        "ask": 100.1,
        "depth": {"buy": [{"price": 99.9}], "sell": [{"price": 100.1}]},
    }


def _emit_live_tick(
    mdm: MarketDataManager, symbol: str, token: int, *, ltp: float, source: str = "ws"
) -> None:
    exchange_ts = datetime.now(timezone.utc).timestamp()
    mdm._emit_tick(  # noqa: SLF001
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": ltp,
            "bid": ltp - 0.1,
            "ask": ltp + 0.1,
            "timestamp": exchange_ts,
            "exchange_timestamp": exchange_ts,
            "source": source,
            "depth": {
                "buy": [{"price": ltp - 0.1}],
                "sell": [{"price": ltp + 0.1}],
            },
        },
        source=source,
    )


def test_cache_rejected_current_generation_ws_ticks_count_for_live_execution() -> None:
    """#1157 fanout ticks must also satisfy the genuine live-tick execution gate."""
    apply_patches()
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    symbol = "NFO:NIFTY26AUG25000PE"
    token = 123
    mdm.register_symbol(symbol, token)
    mdm._selected_pe_symbol = symbol  # noqa: SLF001
    mdm.subscribe(symbol, lambda _tick: None)
    _seed_future_cache(mdm, symbol, token)

    _emit_live_tick(mdm, symbol, token, ltp=101.0)
    _emit_live_tick(mdm, symbol, token, ltp=101.5)

    snapshot = mdm.get_symbol_snapshot(symbol)
    assert snapshot.real_ticks_last_60s >= 2


def test_poll_ticks_do_not_inflate_genuine_ws_receipt_count() -> None:
    """REST/poll freshness must not satisfy the LIVE real-WebSocket-tick proof."""
    apply_patches()
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    symbol = "NFO:NIFTY26AUG25000PE"
    token = 123
    mdm.register_symbol(symbol, token)

    _emit_live_tick(mdm, symbol, token, ltp=101.0, source="poll")
    _emit_live_tick(mdm, symbol, token, ltp=101.5, source="poll")

    snapshot = mdm.get_symbol_snapshot(symbol)
    assert snapshot.real_ticks_last_60s == 0
