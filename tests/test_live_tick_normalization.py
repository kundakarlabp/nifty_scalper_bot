from datetime import datetime, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_ws_raw_tick_normalizes_to_canonical_tick() -> None:
    mdm = MarketDataManager()
    mdm.register_symbol("NSE:NIFTY", 256265)
    tick = mdm.normalize_live_tick(
        {"instrument_token": 256265, "last_price": 22450.5},
        source="ws",
    )
    assert tick is not None
    assert tick["symbol"] == "NSE:NIFTY"
    assert tick["token"] == 256265
    assert tick["ltp"] == 22450.5
    assert tick["source"] == "ws"


def test_missing_bid_ask_stays_none() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick(
        {"symbol": "NSE:NIFTY", "ltp": 22450.0},
        source="ws",
    )
    assert tick is not None
    assert tick["bid"] is None
    assert tick["ask"] is None


def test_ltp_non_positive_returns_none() -> None:
    mdm = MarketDataManager()
    assert (
        mdm.normalize_live_tick(
            {"symbol": "NSE:NIFTY", "ltp": 0},
            source="ws",
        )
        is None
    )


def test_timestamp_is_timezone_aware_utc() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick(
        {
            "symbol": "NSE:NIFTY",
            "ltp": 1.0,
            "timestamp": "2026-01-01T10:00:00",
        },
        source="poll",
    )
    assert tick is not None
    ts = tick["timestamp"]
    assert isinstance(ts, datetime)
    assert ts.tzinfo is not None
    assert ts.tzinfo == timezone.utc


def test_received_at_fallback_preserves_provenance_and_is_not_tradable() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick(
        {
            "symbol": "NFO:NIFTY26MAY24000CE",
            "ltp": 120.0,
            "received_at": "2026-01-01T10:00:00Z",
            "depth": {
                "buy": [{"price": 119.5, "quantity": 100}],
                "sell": [{"price": 120.5, "quantity": 120}],
            },
            "timestamp_source": "received_at",
            "source_timestamp_valid": False,
            "timestamp_quality": "received_at",
        },
        source="ws",
    )

    assert tick is not None
    assert tick["timestamp_source"] == "received_at"
    assert tick["timestamp_quality"] == "received_at"
    assert tick["source_timestamp_valid"] is False
    assert tick["event_timestamp_ms"] is None
    assert MarketDataManager._tick_event_wallclock(tick) is None
    assert tick["hard_readiness_eligible"] is False
    assert tick["tradable_quote"] is False


def test_ws_missing_event_timestamp_is_receive_only_not_event_clock() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick(
        {
            "symbol": "NFO:NIFTY26MAY24000CE",
            "ltp": 120.0,
            "bid": 119.9,
            "ask": 120.1,
        },
        source="ws",
    )

    assert tick is not None
    assert tick["timestamp_quality"] == "synthetic"
    assert tick["source_timestamp_valid"] is False
    assert tick["event_timestamp_ms"] is None
    assert float(tick["received_timestamp_ms"]) > 0.0
    assert int(tick["received_monotonic_ns"]) > 0
    assert tick["tradable_quote"] is False
