from __future__ import annotations

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


SYMBOL = "NFO:NIFTY26SEP25000CE"
TOKEN = 123456


def _tick(
    *,
    received_at: float,
    bid: float = 100.0,
    ask: float = 100.5,
    bid_qty: int = 100,
    ask_qty: int = 100,
    token: int = TOKEN,
    source: str = "ws",
) -> dict[str, object]:
    return {
        "symbol": SYMBOL,
        "token": token,
        "instrument_token": token,
        "source": source,
        "received_at": received_at,
        "bid": bid,
        "ask": ask,
        "best_bid": bid,
        "best_ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "buy_qty": bid_qty,
        "sell_qty": ask_qty,
        "depth": {
            "buy": [{"price": bid, "quantity": bid_qty}],
            "sell": [{"price": ask, "quantity": ask_qty}],
        },
        "depth_available": True,
        "tradable_quote": True,
        "ltp": 100.25,
        "last_price": 100.25,
    }


def test_temporal_ofi_uses_book_events_not_static_depth_snapshot() -> None:
    mdm = MarketDataManager()

    first = mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.0, bid_qty=100, ask_qty=100)
    )
    second = mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.2, bid_qty=140, ask_qty=100)
    )
    third = mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.4, bid_qty=160, ask_qty=100)
    )

    assert first["ofi_ready"] is False
    assert first["ofi_update_count_1s"] == 0
    assert second["ofi_event"] == pytest.approx(40.0)
    assert third["ofi_event"] == pytest.approx(20.0)
    assert third["ofi_1s"] == pytest.approx(60.0)
    assert third["ofi_1s_normalized"] > 0.0
    assert third["ofi_ready"] is True
    assert third["ofi_source"] == "ws_full_depth"


def test_temporal_ofi_resets_on_token_change_and_does_not_cross_contracts() -> None:
    mdm = MarketDataManager()

    mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.0, bid_qty=100, ask_qty=100)
    )
    ready = mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.2, bid_qty=150, ask_qty=100)
    )
    reset = mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.4, bid_qty=200, ask_qty=100, token=TOKEN + 1)
    )

    assert ready["ofi_update_count_1s"] == 1
    assert reset["ofi_ready"] is False
    assert reset["ofi_update_count_1s"] == 0
    assert reset["ofi_1s"] == pytest.approx(0.0)


def test_temporal_ofi_ignores_non_ws_fallback_quotes() -> None:
    mdm = MarketDataManager()

    fallback = mdm._enrich_order_flow_imbalance(
        _tick(received_at=1000.0, source="poll")
    )

    assert "ofi_ready" not in fallback
    assert "ofi_1s_normalized" not in fallback
