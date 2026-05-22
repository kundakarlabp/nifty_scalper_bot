import logging
import os

from nifty_scalper_bot.utils.log_throttle import LogThrottle, log_throttled


def test_same_key_suppresses_within_interval(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("test.same_key")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "EV", "k1", 10.0, "msg", throttle=throttle)
        assert not log_throttled(logger, logging.INFO, "EV", "k1", 10.0, "msg", throttle=throttle)
    assert [r.message for r in caplog.records if r.message == "msg"] == ["msg"]


def test_different_keys_do_not_suppress_each_other(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("test.diff_keys")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "EV", "k1", 10.0, "msg1", throttle=throttle)
        assert log_throttled(logger, logging.INFO, "EV", "k2", 10.0, "msg2", throttle=throttle)
    assert any(r.message == "msg1" for r in caplog.records)
    assert any(r.message == "msg2" for r in caplog.records)


def test_suppressed_count_emitted_on_next_log(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("test.suppressed_count")
    with caplog.at_level(logging.INFO):
        log_throttled(logger, logging.INFO, "EV", "k1", 10.0, "first", throttle=throttle)
        log_throttled(logger, logging.INFO, "EV", "k1", 10.0, "supp", throttle=throttle)
        assert log_throttled(logger, logging.INFO, "EV", "k1", 0.0, "second", throttle=throttle)
    second = [r for r in caplog.records if r.message == "second"][0]
    assert getattr(second, "suppressed_count", 0) == 1


def test_critical_order_events_not_throttled(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("test.critical")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "ORDER_PATH_ENTERED", "order:k1", 0.0, "order path", throttle=throttle)
        assert log_throttled(logger, logging.INFO, "ORDER_PATH_ENTERED", "order:k1", 0.0, "order path", throttle=throttle)
    assert sum(1 for r in caplog.records if r.message == "order path") == 2


def test_verbose_env_flags_default_false():
    assert str(os.getenv("LOG_VERBOSE_TICKS", "false")).strip().lower() not in {"1", "true", "yes", "y", "on"}
    assert str(os.getenv("LOG_VERBOSE_CANDLES", "false")).strip().lower() not in {"1", "true", "yes", "y", "on"}
