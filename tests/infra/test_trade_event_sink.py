from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from nifty_scalper_bot.infra.trade_event_sink import TradeEventSink
from nifty_scalper_bot.infra.trade_observability import TradeEvent

IST = ZoneInfo("Asia/Kolkata")


def test_sink_disabled_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    sink = TradeEventSink()
    event = TradeEvent("signal.generated", datetime.now(IST))
    assert sink.emit(event) is False


def test_sink_queue_saturation_fails_open(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    sink = TradeEventSink(queue_size=1)
    sink.start = Mock()  # type: ignore[method-assign]
    assert sink.emit(TradeEvent("signal.generated", datetime.now(IST))) is True
    assert sink.emit(TradeEvent("candidate.approved", datetime.now(IST))) is False
    assert sink.dropped == 1


def test_persist_uses_trade_events_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    post = Mock(return_value=response)
    monkeypatch.setattr(
        "nifty_scalper_bot.infra.trade_event_sink.requests.post",
        post,
    )
    sink = TradeEventSink()
    sink._persist(
        TradeEvent(
            "order.submitted",
            datetime.now(IST),
            trade_id="trade-1",
            trace_id="trace-1",
        )
    )
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/rest/v1/nifty_trade_events")
