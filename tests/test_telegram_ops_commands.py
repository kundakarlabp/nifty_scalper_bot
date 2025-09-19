from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from src.notifications.telegram_commands import TelegramCommands


def test_ops_commands_adjust_state(monkeypatch) -> None:
    sent: list[str] = []
    tc = TelegramCommands("t", "1")
    tc._send = lambda msg: sent.append(msg)

    tc._handle_cmd("/risk", "1.2%")
    assert tc.risk_pct == 1.2 and "risk_pct" in sent[-1]

    tc._handle_cmd("/exposure", "underlying")
    assert tc.exposure_mode == "underlying"

    start = time.time()
    tc._handle_cmd("/pause", "15m")
    assert tc.paused_until >= start + 15 * 60 - 1

    tc.lots = 5
    tc.unit_notional = 100.0
    tc._handle_cmd("/flatten", "")
    assert tc.lots == 0 and tc.unit_notional == 0.0

    tc._handle_cmd("/status", "brief")
    msg = sent[-1]
    assert "basis=" in msg and "lots=" in msg


def _stub_settings() -> SimpleNamespace:
    return SimpleNamespace(
        LOG_DIAG_DEFAULT_SEC=10,
        LOG_TRACE_DEFAULT_SEC=5,
        LOG_MAX_LINES_REPLY=8,
    )


def test_logs_command_controls_windows(monkeypatch) -> None:
    sent: list[str] = []
    tc = TelegramCommands("t", "1", settings_obj=_stub_settings())
    tc._send = sent.append

    monkeypatch.setattr(time, "time", lambda: 100.0)
    assert not tc._window_active("diag")
    tc._handle_cmd("/logs", "on 15")
    assert pytest.approx(tc._diag_until_ts, rel=0.0) == 115.0
    assert "15s" in sent[-1]
    assert tc._window_active("diag")

    monkeypatch.setattr(time, "time", lambda: 200.0)
    assert not tc._window_active("diag")

    tc._handle_cmd("/logs", "off")
    assert sent[-1].startswith("✅ Logging window")
    assert tc._diag_until_ts == 0.0 and tc._trace_until_ts == 0.0


def test_trace_command_uses_default(monkeypatch) -> None:
    sent: list[str] = []
    tc = TelegramCommands("t", "1", settings_obj=_stub_settings())
    tc._send = sent.append

    monkeypatch.setattr(time, "time", lambda: 50.0)
    tc._handle_cmd("/trace", "")

    assert pytest.approx(tc._trace_until_ts, rel=0.0) == 55.0
    assert "TRACE window enabled" in sent[-1]


def test_diag_command_formats_snapshot(monkeypatch) -> None:
    sent: list[str] = []
    tc = TelegramCommands("t", "1", settings_obj=_stub_settings())
    tc._send = sent.append

    monkeypatch.setattr(
        "src.diagnostics.healthkit.snapshot_pipeline",
        lambda: {"loop": {"ticks": 1}, "health": {"ok": True}},
    )

    handled = tc._handle_cmd("/diag", "")
    assert handled is True
    assert sent
    assert "ticks" in sent[-1]
    assert sent[-1].startswith("```json")


def test_micro_command_handles_source(monkeypatch) -> None:
    class DummySource:
        def current_tokens(self) -> tuple[int, int]:
            return 101, 202

        def get_micro_state(self, token: int | None) -> dict[str, int | None]:
            return {"token": token}

    sent: list[str] = []
    tc = TelegramCommands("t", "1", settings_obj=_stub_settings(), source=DummySource())
    tc._send = sent.append

    handled = tc._handle_cmd("/micro", "")
    assert handled is True
    assert "101" in sent[-1] and "202" in sent[-1]


def test_quotes_command_handles_source() -> None:
    class DummySource:
        def current_tokens(self) -> tuple[int, int]:
            return 111, 222

        def quote_snapshot(self, token: int | None) -> dict[str, int | None]:
            return {"token": token, "bid": 10, "ask": 12}

    sent: list[str] = []
    tc = TelegramCommands("t", "1", settings_obj=_stub_settings(), source=DummySource())
    tc._send = sent.append

    handled = tc._handle_cmd("/quotes", "")
    assert handled is True
    assert "111" in sent[-1] and "12" in sent[-1]
