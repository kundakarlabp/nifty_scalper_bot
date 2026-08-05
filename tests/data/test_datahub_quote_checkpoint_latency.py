from __future__ import annotations

import threading
from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub


class _MDM:
    def attach_tick_bus(self, _bus: Any) -> None:
        return None


class _Store:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, Any], int]] = []

    def load_snapshot(self, _kind: str) -> dict[str, Any]:
        return {}

    def save_snapshot(self, kind: str, payload: dict[str, Any]) -> None:
        self.saved.append((kind, payload, threading.get_ident()))


def test_quote_snapshot_copy_runs_on_persistence_worker() -> None:
    store = _Store()
    hub = DataHub(_MDM(), store=store)
    caller_thread = threading.get_ident()
    snapshot_threads: list[int] = []
    original = hub._snapshot_quotes_unlocked

    def _record_snapshot_thread() -> dict[str, Any]:
        snapshot_threads.append(threading.get_ident())
        return original()

    hub._snapshot_quotes_unlocked = _record_snapshot_thread  # type: ignore[method-assign]
    with hub._lock:
        hub._quotes["NFO:NIFTY26AUG25000CE"] = {"ltp": 100.0}
        hub._quote_checkpoint_dirty = True
        hub._quote_checkpoint_deadline = 0.0

    hub.checkpoint_quotes(force=True, wait=True)

    assert snapshot_threads
    assert all(thread_id != caller_thread for thread_id in snapshot_threads)
    assert store.saved[0][0] == "quotes"
    assert store.saved[0][1]["NFO:NIFTY26AUG25000CE"]["ltp"] == 100.0
    hub.close()


def test_quote_checkpoint_enqueues_lazy_snapshot_request() -> None:
    store = _Store()
    hub = DataHub(_MDM(), store=store)
    entered = threading.Event()
    release = threading.Event()
    original = hub._snapshot_quotes_unlocked

    def _blocked_snapshot() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=5.0)
        return original()

    hub._snapshot_quotes_unlocked = _blocked_snapshot  # type: ignore[method-assign]
    with hub._lock:
        hub._quotes["NSE:NIFTY"] = {"ltp": 25000.0}
        hub._quote_checkpoint_dirty = True
        hub._quote_checkpoint_deadline = 0.0

    hub.checkpoint_quotes(force=True)

    assert entered.wait(timeout=5.0)
    # The caller returned even though snapshot construction is still blocked on
    # the persistence worker. This is the tick-path latency contract.
    assert hub.quote_checkpoint_inflight() is True
    release.set()
    assert hub.flush_persistence(timeout=5.0) is True
    hub.close()
