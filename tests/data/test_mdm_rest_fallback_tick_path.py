from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager

SYMBOL = "NSE:NIFTY"
TOKEN = 256265
TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _manager(*, token: bool = True) -> MarketDataManager:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    if token:
        mdm.register_symbol(SYMBOL, TOKEN)
    return mdm


def test_ingest_rest_quote_without_token_enters_candle_engine_once() -> None:
    mdm = _manager(token=False)
    engine = mdm.get_candle_engine(SYMBOL)
    assert not hasattr(mdm, "_process_poll_quote")
    mdm._enqueue_tick_threadsafe = lambda tick: mdm._process_queued_tick(tick)  # type: ignore[method-assign]

    mdm.ingest_rest_quote(SYMBOL, {"last_price": 100.0, "timestamp": TS})

    assert engine.current_candle is not None
    assert engine.current_candle["close"] == 100.0
    latest_tick = mdm.get_latest_tick(SYMBOL)
    assert latest_tick is not None
    assert latest_tick["timestamp_source"] == "rest_poll"
    assert engine.get_completed_bars() == []
    assert mdm._candle_metrics["fallback_quote_tick_total"] == 1


def test_equivalent_ws_and_rest_quote_mutates_candle_once() -> None:
    mdm = _manager(token=True)
    engine = mdm.get_candle_engine(SYMBOL)
    mdm._enqueue_tick_threadsafe = lambda tick: mdm._process_queued_tick(tick)  # type: ignore[method-assign]

    mdm._process_queued_tick(
        {
            "instrument_token": TOKEN,
            "symbol": SYMBOL,
            "last_price": 100.0,
            "exchange_timestamp": TS,
        }
    )
    first_candle = dict(engine.current_candle or {})
    mdm.ingest_rest_quote(SYMBOL, {"last_price": 100.0, "timestamp": TS})

    assert engine.current_candle == first_candle
    assert engine.get_completed_bars() == []
