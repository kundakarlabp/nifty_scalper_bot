from __future__ import annotations

"""Tests for the structured logging helpers."""

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.logs import StructuredLogger, _ensure_sections, _normalize


def test_normalize_handles_collections_and_custom_numeric() -> None:
    class Convertible:
        def __int__(self) -> int:  # pragma: no cover - trivial method
            return 7

    data = {
        "mapping": {"x": Convertible()},
        "sequence": [Convertible(), None],
        "custom": Convertible(),
        "fallback": object(),
    }

    normalized = _normalize(data)

    assert normalized["mapping"] == {"x": 7}
    assert normalized["sequence"] == [7, None]
    assert normalized["custom"] == 7
    assert isinstance(normalized["fallback"], str)


def test_ensure_sections_populates_expected_keys() -> None:
    payload: dict[str, Any] = {"data": "raw"}

    _ensure_sections(payload)

    assert set(payload["data"].keys()) == {
        "subscribe",
        "tokens_resolved",
        "tick_full",
        "tick_ltp_only",
        "value",
    }
    assert payload["data"]["subscribe"] is None
    assert payload["data"]["value"] == "raw"
    assert set(payload["signal"].keys()) == {"block_micro", "block_size"}
    assert set(payload["trade"].keys()) == {"ctx"}


def test_structured_logger_event_merges_defaults(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="structured")

    logger = StructuredLogger(defaults={"base": 1})
    bound = logger.bind(extra="value")

    bound.event("trade_executed", qty=2)

    payload = json.loads(caplog.records[-1].getMessage())
    assert payload["event"] == "trade_executed"
    assert payload["base"] == 1
    assert payload["extra"] == "value"
    assert payload["qty"] == 2


def test_structured_logger_trace_gated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="structured")

    # Mock the optional healthkit module with trace inactive.
    module = SimpleNamespace(trace_active=lambda: False)
    monkeypatch.setattr("src.logs.import_module", lambda name: module)

    logger = StructuredLogger()
    logger.event("micro_eval", status="ok")

    debug_messages = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert debug_messages, "expected debug call when trace inactive"
    payload = json.loads(debug_messages[-1])
    assert payload["event"] == "micro_eval"
    assert payload["status"] == "ok"
