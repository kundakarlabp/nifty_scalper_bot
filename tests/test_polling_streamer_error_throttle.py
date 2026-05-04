from nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


def test_log_quote_request_failed_throttles():
    p = PollingStreamer(broker_client=None, on_tick=lambda *_: None, instrument_resolver=None)
    p._log_quote_request_failed(['NSE:NIFTY'], RuntimeError('x'))
    before = dict(p._last_poll_error_log)
    p._log_quote_request_failed(['NSE:NIFTY'], RuntimeError('x'))
    assert p._last_poll_error_log == before
