from __future__ import annotations

"""Unit tests for logging setup helpers."""

import logging
from typing import Any

import pytest

from src.server import logging_setup as ls


def test_q_formats_values() -> None:
    assert ls._q(None) == "null"
    assert ls._q(True) == "true"
    assert ls._q(3.14) == "3.14"
    assert ls._q("hello world") == '"hello world"'


def test_collect_extra_merges_custom_fields() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="example",
        args=(),
        exc_info=None,
    )
    record.extra = {"foo": "bar"}
    record.custom = 1

    extra = ls._collect_extra(record)

    assert extra["foo"] == "bar"
    assert extra["custom"] == 1


def test_logfmt_formatter_includes_extra_fields() -> None:
    formatter = ls._LogfmtFormatter()
    record = logging.LogRecord(
        name="logfmt",
        level=logging.WARNING,
        pathname=__file__,
        lineno=20,
        msg="alert triggered",
        args=(),
        exc_info=None,
    )
    record.extra = {"extra": {"source": "unit"}}
    record.tag = "alert"

    formatted = formatter.format(record)

    assert "tag=alert" in formatted
    assert "msg=\"alert triggered\"" in formatted
    assert "extra=\"{'source': 'unit'}\"" in formatted


def test_json_formatter_serializes_payload() -> None:
    formatter = ls._JsonFormatter()
    record = logging.LogRecord(
        name="json",
        level=logging.ERROR,
        pathname=__file__,
        lineno=30,
        msg="failure",
        args=(),
        exc_info=None,
    )
    record.extra = {"count": 2}

    payload = formatter.format(record)

    assert '"msg": "failure"' in payload
    assert '"count": 2' in payload


def test_log_event_respects_requested_level(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="event.tag")

    ls.log_event("event.tag", level="warning", detail="value")

    record = caplog.records[-1]
    assert record.levelname == "WARNING"
    assert record.tag == "event.tag"
    assert record.detail == "value"


def test_log_suppressor_tracks_repeats(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    suppressor = ls.LogSuppressor(window_sec=0.01)

    times: list[float] = [0.02, 0.025, 0.05]

    def fake_time() -> float:
        return times.pop(0) if times else 0.05

    monkeypatch.setattr("time.time", fake_time)

    assert suppressor.should_log("warn", "group", 1) is True
    # Second call within the window should be suppressed.
    assert suppressor.should_log("warn", "group", 1) is False
    # Third call after window should emit and report the repeat.
    assert suppressor.should_log("warn", "group", 1) is True

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert warnings[-1].repeats == 1
