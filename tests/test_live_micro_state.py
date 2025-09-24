from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.data.source import LiveKiteSource


class Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def time(self) -> float:
        return self.now


@pytest.fixture()
def patched_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Gate:
        def should_emit(self, key: str, force: bool = False) -> bool:
            return True

    settings_stub = SimpleNamespace(
        micro=SimpleNamespace(depth_min_lots=1, max_spread_pct=5.0),
        instruments=SimpleNamespace(nifty_lot_size=75),
        TICK_MAX_LAG_S=5.0,
        TICK_STALE_SECONDS=0.75,
        build_log_gate=lambda interval_s=None: _Gate(),
    )
    monkeypatch.setattr("src.data.source.settings", settings_stub, raising=False)


def _make_full_tick(token: int = 111) -> dict[str, object]:
    return {
        "instrument_token": token,
        "depth": {
            "buy": [{"price": 100.0, "quantity": 150}],
            "sell": [{"price": 101.0, "quantity": 200}],
        },
    }


def test_micro_state_updates_on_full_tick(
    patched_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(1_000.0)
    monkeypatch.setattr("src.data.source.time.time", clock.time)
    src = LiveKiteSource(kite=None)
    src._on_tick(_make_full_tick())

    state = src.get_micro_state(111)
    assert state["stale"] is False
    assert state["has_depth"] is True
    assert state["depth_ok"] is True
    assert state["spread_pct"] is not None
    assert state["bid"] == pytest.approx(100.0)
    assert state["ask"] == pytest.approx(101.0)
    assert state["bid_qty"] == 150
    assert state["ask_qty"] == 200


def test_micro_state_marks_stale_once(
    patched_settings: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, LiveKiteSource.__name__)
    clock = Clock(2_000.0)
    monkeypatch.setattr("src.data.source.time.time", clock.time)
    src = LiveKiteSource(kite=None)
    src._on_tick(_make_full_tick(token=222))

    clock.now += 2.0
    state = src.get_micro_state(222)
    assert state["stale"] is True
    assert any(record.message == "data.tick_stale" for record in caplog.records)


def test_quote_snapshot_returns_latest_depth(
    patched_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock(5_000.0)
    monkeypatch.setattr("src.data.source.time.time", clock.time)
    src = LiveKiteSource(kite=None)
    src._on_tick(_make_full_tick(token=333))

    snap = src.quote_snapshot(333)
    assert snap is not None
    assert snap["bid"] == pytest.approx(100.0)
    assert snap["ask_qty"] == 200
    assert snap["has_depth"] is True
    assert snap["age_sec"] == pytest.approx(0.0)
    assert src.quote_snapshot(99999) is None

