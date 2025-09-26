from __future__ import annotations

import time

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
        EXPOSURE_CAP_PCT=5.0,
    )

    class DummySource:
        def __init__(self) -> None:
            self.atm_tokens = (111, 222)
            self._current_tokens: tuple[int, int] | None = None
            self._force_reconnects = 0

        def current_tokens(self) -> tuple[int, int] | None:
            return self._current_tokens

        def get_micro_state(self, token: int) -> dict[str, object]:
            phase = "after" if self._current_tokens else "before"
            return {"token": token, "depth_ok": 1, "phase": phase}

        def quote_snapshot(self, token: int) -> dict[str, object]:
            phase = "after" if self._current_tokens else "before"
            return {
                "token": token,
                "bid": token + 1,
                "ask": token + 2,
                "phase": phase,
            }

        def subscription_modes(self) -> dict[int, str]:
            tokens = self._current_tokens or self.atm_tokens
            return {tok: "FULL" for tok in tokens}

        def ws_diag_snapshot(self) -> dict[str, object]:
            return {
                "connected": True,
                "subs_count": len(self.subscription_modes()),
                "last_tick_age_ms": 1234,
            }

        def force_hard_reconnect(self) -> None:
            self._force_reconnects += 1

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

    assert tc._handle_cmd("/diag", "") is True
    assert sent[-1].startswith("WS Connected: True")
    snapshot = tc._diag_snapshot()
    assert snapshot["risk"]["cap_pct"] == 5.0

    assert tc._handle_cmd("/micro", "") is True
    micro_before = sent[-1]
    assert "\"CE\"" in micro_before
    assert "\"phase\": \"before\"" in micro_before

    assert tc._handle_cmd("/quotes", "") is True
    quotes_before = sent[-1]
    assert "bid" in quotes_before
    assert "\"phase\": \"before\"" in quotes_before

    source._current_tokens = (333, 444)

    assert tc._handle_cmd("/micro", "") is True
    micro_after = sent[-1]
    assert "\"CE\"" in micro_after
    assert "\"phase\": \"after\"" in micro_after

    assert tc._handle_cmd("/quotes", "") is True
    quotes_after = sent[-1]
    assert "bid" in quotes_after
    assert "\"phase\": \"after\"" in quotes_after

    assert tc._handle_cmd("/subs", "") is True
    assert "Subscriptions:" in sent[-1]

    assert tc._handle_cmd("/fresh", "") is True
    assert sent[-1] == "Hard reconnect initiated"
    assert source._force_reconnects == 1


def test_diag_with_source_only() -> None:
    dummy_settings = SimpleNamespace(
        LOG_MAX_LINES_REPLY=10,
        EXPOSURE_CAP_PCT=2.5,
    )

    class DummySource:
        def current_tokens(self) -> tuple[int, int]:
            return (111, 0)

        def get_micro_state(self, token: int) -> dict[str, int]:
            return {"token": token, "depth_ok": 1}

    tc = TelegramCommands("t", "1", settings=dummy_settings, source=DummySource())
    sent: list[str] = []
    tc._send = lambda msg: sent.append(msg)

    assert tc._handle_cmd("/diag", "") is True
    assert sent, "Expected /diag to produce output"
    text = sent[-1]
    assert text == "Diag: Legacy build"
    snapshot = tc._diag_snapshot()
    assert snapshot["risk"]["cap_pct"] == 2.5
    assert snapshot["micro"]["ce"]["token"] == 111
