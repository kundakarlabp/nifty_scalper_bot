from __future__ import annotations

import logging
import sys
from contextlib import contextmanager

import pytest

from src.utils import logger_setup


@contextmanager
def _preserve_root_logger():
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        root.handlers.clear()
        yield root
    finally:
        root.handlers.clear()
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)


def _stream_handler_from(root: logging.Logger) -> logging.Handler:
    for handler in root.handlers:
        if getattr(handler, "stream", None) is sys.stdout:
            return handler
    raise AssertionError("stream handler bound to sys.stdout not found")


def test_setup_logging_respects_log_format_logfmt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_FORMAT", "logfmt")
    monkeypatch.delenv("LOG_JSON", raising=False)
    with _preserve_root_logger() as root:
        logger_setup.setup_logging()
        handler = _stream_handler_from(root)
        assert isinstance(handler.formatter, logger_setup._LineFormatter)


def test_setup_logging_respects_log_format_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.delenv("LOG_JSON", raising=False)
    with _preserve_root_logger() as root:
        logger_setup.setup_logging()
        handler = _stream_handler_from(root)
        assert isinstance(handler.formatter, logger_setup._JsonFormatter)


def test_setup_logging_falls_back_to_log_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_FORMAT", "invalid")
    monkeypatch.setenv("LOG_JSON", "false")
    with _preserve_root_logger() as root:
        logger_setup.setup_logging()
        handler = _stream_handler_from(root)
        assert isinstance(handler.formatter, logger_setup._LineFormatter)


def test_setup_logging_defaults_to_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_JSON", raising=False)
    with _preserve_root_logger() as root:
        logger_setup.setup_logging()
        handler = _stream_handler_from(root)
        assert isinstance(handler.formatter, logger_setup._JsonFormatter)
