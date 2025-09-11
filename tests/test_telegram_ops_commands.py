from __future__ import annotations

import time

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
