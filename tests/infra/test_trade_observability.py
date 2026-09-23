from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from nifty_scalper_bot.infra.trade_observability import TradeEvent

IST = ZoneInfo("Asia/Kolkata")


def test_trade_event_row_has_canonical_trading_date() -> None:
    event = TradeEvent(
        event_name="signal.generated",
        event_at=datetime(2026, 9, 23, 9, 16, tzinfo=IST),
        signal_id="sig-1",
        symbol="NFO:NIFTYCE",
        payload={"score": 78.0},
    )

    row = event.row()

    assert row["trading_date"] == "2026-09-23"
    assert row["event_name"] == "signal.generated"
    assert row["signal_id"] == "sig-1"
    assert row["payload"]["score"] == 78.0


def test_trade_event_preserves_end_to_end_correlation() -> None:
    event = TradeEvent(
        event_name="entry.filled",
        event_at=datetime(2026, 9, 23, 10, 17, tzinfo=IST),
        trade_id="trade-1",
        signal_id="signal-1",
        trace_id="trace-1",
    )

    row = event.row()

    assert row["trade_id"] == "trade-1"
    assert row["signal_id"] == "signal-1"
    assert row["trace_id"] == "trace-1"
