from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "event_buffer.py"
SPEC = importlib.util.spec_from_file_location("streamlit_event_buffer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_trade_event_is_kept() -> None:
    event = MODULE.parse_event("[2026-06-23 09:35:00 IST] ORDER_SENT NIFTY CE")
    assert event is not None
    assert event["type"] == "TRADE"


def test_heartbeat_is_dropped() -> None:
    assert MODULE.parse_event("[2026-06-23 09:35:00 IST] HEARTBEAT ok") is None


def test_signal_event_is_classified() -> None:
    event = MODULE.parse_event("[2026-06-23 09:35:00 IST] SIGNAL_GENERATED BUY")
    assert event is not None
    assert event["type"] == "SIGNAL"
