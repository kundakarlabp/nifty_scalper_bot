import logging

import pytest

from src.server import logging_setup


def test_log_event_suppression(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("LOG_SUPPRESS_WINDOW_SEC", "120")
    logging_setup._reset_log_event_suppressor()

    with caplog.at_level(logging.INFO):
        logging_setup.log_event("test.tag", "info", foo="bar", status="ok")
        logging_setup.log_event("test.tag", "info", foo="bar", status="ok")

    first_batch = [rec for rec in caplog.records if rec.name == "test.tag"]
    assert len(first_batch) == 1
    first_payload = getattr(first_batch[0], "extra", {})
    assert isinstance(first_payload, dict)
    assert first_payload.get("foo") == "bar"

    caplog.clear()

    with caplog.at_level(logging.INFO):
        logging_setup.log_event("test.tag", "info", foo="baz", status="ok")

    second_batch = [rec for rec in caplog.records if rec.name == "test.tag"]
    assert len(second_batch) == 1
    second_payload = getattr(second_batch[0], "extra", {})
    assert isinstance(second_payload, dict)
    assert second_payload.get("foo") == "baz"
