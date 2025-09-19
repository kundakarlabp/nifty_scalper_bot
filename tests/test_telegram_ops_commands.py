from __future__ import annotations

import time

import sys
from types import SimpleNamespace

import src.notifications.telegram_commands as tg_mod
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


def test_debug_windows_and_diagnostics(monkeypatch) -> None:
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(tg_mod.time, "time", lambda: fake_now["value"])

    dummy_settings = SimpleNamespace(
        LOG_DIAG_DEFAULT_SEC=5,
        LOG_TRACE_DEFAULT_SEC=7,
        LOG_MAX_LINES_REPLY=10,
    )

    class DummySource:
        def current_tokens(self) -> tuple[int, int]:
            return (111, 222)

        def get_micro_state(self, token: int) -> dict[str, int]:
            return {"token": token, "depth_ok": 1}

        def quote_snapshot(self, token: int) -> dict[str, int]:
            return {"token": token, "bid": token + 1, "ask": token + 2}

    source = DummySource()
    tc = TelegramCommands("t", "1", settings=dummy_settings, source=source)
    sent: list[str] = []
    tc._send = lambda msg: sent.append(msg)

    assert tc._handle_cmd("/logs", "on 4") is True
    assert tc._window_active("diag") is True
    fake_now["value"] += 5.0
    assert tc._window_active("diag") is False

    assert tc._handle_cmd("/trace", "3") is True
    assert tc._window_active("trace") is True
    fake_now["value"] += 4.0
    assert tc._window_active("trace") is False

    assert tc._handle_cmd("/logs", "off") is True
    assert tc._diag_until_ts == 0.0
    assert tc._trace_until_ts == 0.0

    dummy_diag = SimpleNamespace(snapshot_pipeline=lambda: {"status": "ok"})
    monkeypatch.setitem(sys.modules, "src.diagnostics.healthkit", dummy_diag)
    assert tc._handle_cmd("/diag", "") is True
    assert sent[-1].startswith("```")
    assert "status" in sent[-1]

    assert tc._handle_cmd("/micro", "") is True
    assert "\"CE\"" in sent[-1]

    assert tc._handle_cmd("/quotes", "") is True
    assert "bid" in sent[-1]
