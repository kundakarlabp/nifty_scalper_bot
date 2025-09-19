"""Ensure structured logger payloads expose deployment diagnostics keys."""

from __future__ import annotations

import json

from src.logs import StructuredLogger


def test_structured_logger_injects_required_sections(monkeypatch) -> None:
    messages: list[str] = []

    class DummyLogger:
        def info(self, message: str) -> None:  # pragma: no cover - exercised via call
            messages.append(message)

    logger = StructuredLogger(name="structured_test")
    monkeypatch.setattr(logger, "_logger", DummyLogger(), raising=False)

    logger.event(
        "runner_flow",
        data={"subscribe": True},
        signal={"block_micro": "depth"},
    )

    assert messages, "structured logger did not emit payload"
    payload = json.loads(messages[-1])

    assert payload["data"]["subscribe"] is True
    assert payload["data"]["tokens_resolved"] is None
    assert payload["data"]["tick_full"] is None
    assert payload["data"]["tick_ltp_only"] is None
    assert payload["signal"]["block_micro"] == "depth"
    assert payload["signal"]["block_size"] is None
    assert payload["trade"]["ctx"] is None
