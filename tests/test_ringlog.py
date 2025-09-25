"""Tests for the diagnostics ring buffer helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils import ringlog


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = SimpleNamespace(
        LOG_RING_ENABLED=True,
        log_ring_enabled=True,
        DIAG_RING_SIZE=3,
        diag_ring_size=3,
    )
    monkeypatch.setattr(ringlog, "settings", dummy, raising=False)
    ringlog.clear()


def test_ringlog_append_and_tail() -> None:
    ringlog.append({"id": 1})
    ringlog.append({"id": 2})
    ringlog.append({"id": 3})
    ringlog.append({"id": 4})  # should evict the oldest due to capacity

    entries = ringlog.tail()
    assert len(entries) == 3
    assert entries[0]["id"] == 2
    assert entries[-1]["id"] == 4
    assert ringlog.capacity() == 3


def test_ringlog_clear_and_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ringlog.append({"msg": "hello"})
    assert ringlog.tail()  # buffer has data

    ringlog.clear()
    assert ringlog.tail() == []

    dummy_disabled = SimpleNamespace(
        LOG_RING_ENABLED=False,
        log_ring_enabled=False,
        DIAG_RING_SIZE=2,
        diag_ring_size=2,
    )
    monkeypatch.setattr(ringlog, "settings", dummy_disabled, raising=False)
    ringlog.clear()
    ringlog.append({"msg": "ignored"})
    assert ringlog.tail() == []
