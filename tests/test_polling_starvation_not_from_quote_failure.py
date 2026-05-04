from nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


def test_quote_failure_not_starvation_with_recent_ticks():
    p = PollingStreamer(broker_client=None, on_tick=lambda *_: None, instrument_resolver=None)
    p._last_fetch_status = 'quote_request_failed'
    p._has_recent_tick = lambda: True
    assert p._should_escalate_starvation(tokens_present=True, seen_any_tick=False, ws_healthy=False) is False
