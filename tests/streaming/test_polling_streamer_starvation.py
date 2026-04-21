from src.nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


class _Resolver:
    @staticmethod
    def format_token_as_symbol(_token: int) -> str:
        return "NSE:NIFTY"


class _QuoteBroker:
    def __init__(self, payload):
        self._payload = payload

    def quote(self, _symbols):
        return self._payload


def test_fetch_ticks_marks_empty_quote_map_not_transport_failure() -> None:
    streamer = PollingStreamer(
        broker_client=_QuoteBroker({}),
        on_tick=lambda _tick: None,
        instrument_resolver=_Resolver(),
    )

    ticks = streamer._fetch_ticks([256265])

    assert ticks == []
    assert streamer._last_fetch_status == "empty_quote_map"
    assert (
        streamer._should_escalate_starvation(
            tokens_present=True,
            seen_any_tick=False,
            ws_healthy=False,
        )
        is False
    )


def test_fetch_ticks_marks_quote_failure_as_transport_starvation_candidate() -> None:
    class _BrokenQuoteBroker:
        def quote(self, _symbols):
            raise RuntimeError("network down")

    streamer = PollingStreamer(
        broker_client=_BrokenQuoteBroker(),
        on_tick=lambda _tick: None,
        instrument_resolver=_Resolver(),
    )

    ticks = streamer._fetch_ticks([256265])

    assert ticks == []
    assert streamer._last_fetch_status == "quote_request_failed"
    assert (
        streamer._should_escalate_starvation(
            tokens_present=True,
            seen_any_tick=False,
            ws_healthy=False,
        )
        is True
    )


def test_fetch_ticks_parser_zero_is_not_starvation_candidate() -> None:
    streamer = PollingStreamer(
        broker_client=_QuoteBroker({"NSE:NIFTY": {"last_price": 0}}),
        on_tick=lambda _tick: None,
        instrument_resolver=_Resolver(),
    )

    ticks = streamer._fetch_ticks([256265])

    assert ticks == []
    assert streamer._last_fetch_status == "parser_zero_ticks"
    assert (
        streamer._should_escalate_starvation(
            tokens_present=True,
            seen_any_tick=False,
            ws_healthy=False,
        )
        is False
    )
