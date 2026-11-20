"""Pre-queue WS receipt fidelity and non-invasive MDM capture wiring."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import nifty_scalper_bot.config.paths as config_paths
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.storage.replay_archive import verify_raw_ingress_session


def _capture_owner():
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._replay_capture_lock = threading.RLock()
    mdm._raw_ingress_archive = None
    mdm._raw_ingress_sequence = {}
    mdm._raw_ingress_count_by_day = {}
    mdm._raw_ingress_capture_errors = 0
    mdm._raw_ingress_init_retry_at = 0.0
    mdm._raw_ws_receipts_total = 0
    mdm._logger = logging.getLogger("test_raw_wire")
    mdm._resolve_tick_symbol_for_priority = lambda _raw: "NFO:NIFTYOPT"
    mdm._generation_for_raw_ingress = lambda *_a: 5
    return mdm


def test_raw_capture_precedes_lossy_enqueue_callback(monkeypatch) -> None:
    mdm = _capture_owner()
    mdm._main_loop = SimpleNamespace(is_running=lambda: True)
    observed = []
    mdm._record_raw_ws_receive = lambda *_a: observed.append("receipt")
    mdm._capture_raw_ws_ingress = lambda *_a: observed.append("capture")
    mdm._enqueue_latest_tick_for_drain = lambda *_a: observed.append("queue")
    mdm._enqueue_tick_threadsafe({"instrument_token": 123})
    assert observed == ["receipt", "capture", "queue"]


def test_raw_wire_capture_preserves_original_nested_depth_and_epoch(
    monkeypatch, tmp_path: Path
) -> None:
    import shutil

    mdm = _capture_owner()
    monkeypatch.setenv("REPLAY_RAW_INGRESS_ENABLED", "true")
    monkeypatch.setenv("REPLAY_RAW_MAX_SESSION_MB", "128")
    monkeypatch.setattr(config_paths, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024**3),
    )
    wire = {
        "symbol": "NFO:NIFTYOPT",
        "instrument_token": 7,
        "last_price": 120.25,
        "depth": {"buy": [{"price": 120.2, "quantity": 65}]},
    }
    mdm._raw_ws_receipts_total = 1
    mdm._capture_raw_ws_ingress(wire)
    wire["depth"]["buy"][0]["price"] = 9999
    mdm._raw_ws_receipts_total = 2
    mdm._capture_raw_ws_ingress(wire)
    archive = mdm._raw_ingress_archive
    assert archive is not None
    archive.close()

    from datetime import datetime
    from zoneinfo import ZoneInfo

    day = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    proof = verify_raw_ingress_session(tmp_path / "replay_raw_ingress" / f"{day}.jsonl")
    assert proof["verified"] is True
    assert proof["wire_events"] == 2
    assert mdm.get_raw_ingress_capture_stats()["received_by_symbol_epoch"] == {
        "NFO:NIFTYOPT|7|5": 2
    }
    from nifty_scalper_bot.storage.replay_archive import load_session

    captured = load_session(tmp_path / "replay_raw_ingress" / f"{day}.jsonl")
    assert captured[1]["payload"]["wire"]["depth"]["buy"][0]["price"] == 120.2
    assert captured[2]["payload"]["wire"]["depth"]["buy"][0]["price"] == 9999
