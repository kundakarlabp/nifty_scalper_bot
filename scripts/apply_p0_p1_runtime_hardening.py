from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, *, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


# ---------------------------------------------------------------------------
# P0: fallback tick worker must have one owner and fail closed on stop timeout.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/data/market_data_hardening.py"
text = read(path)
text = replace_once(
    text,
    """    if not isinstance(getattr(self, \"_tick_worker_stop\", None), threading.Event):\n        self._tick_worker_stop = threading.Event()\n    self._candle_flush_task: asyncio.Task[None] | None = None\n""",
    """    if not isinstance(getattr(self, \"_tick_worker_stop\", None), threading.Event):\n        self._tick_worker_stop = threading.Event()\n    self._tick_worker_lifecycle_lock = threading.RLock()\n    self._tick_worker_state = \"STOPPED\"\n    self._tick_worker_generation = 0\n    self._tick_worker_failed_to_stop = False\n    self._tick_worker_stop_timeout_s = _float_env(\n        \"MDM_TICK_WORKER_STOP_TIMEOUT_SECONDS\", 2.0, minimum=0.05\n    )\n    self._candle_flush_task: asyncio.Task[None] | None = None\n""",
    label="worker lifecycle initialization",
)
text = replace_once(
    text,
    """    self._ensure_tick_worker()\n    if not self._put_fallback_tick_nowait(payload):\n""",
    """    if not self._ensure_tick_worker():\n        self._tick_queue_dropped += 1\n        self._record_tick_drop(\n            str(payload.get(\"_mdm_priority_bucket\") or \"unknown\"),\n            \"fallback_worker_unavailable\",\n        )\n        self._logger.error(\n            \"MDM_FALLBACK_TICK_REJECTED reason=worker_unavailable state=%s\",\n            getattr(self, \"_tick_worker_state\", \"unknown\"),\n            extra={\n                \"event\": \"MDM_FALLBACK_TICK_REJECTED\",\n                \"reason\": \"worker_unavailable\",\n                \"worker_state\": getattr(self, \"_tick_worker_state\", \"unknown\"),\n            },\n        )\n        return\n    if not self._put_fallback_tick_nowait(payload):\n""",
    label="worker availability gate",
)
text = regex_once(
    text,
    r"def _ensure_tick_worker_thread_queue\(self: Any\) -> None:\n.*?\n\ndef _safe_queue_task_done",
    dedent(
        '''\
        def _ensure_tick_worker_thread_queue(self: Any) -> bool:
            """Start exactly one fallback worker; never replace a live owner."""
            if not isinstance(getattr(self, "_fallback_tick_queue", None), queue.Queue):
                maxsize = max(
                    int(getattr(self, "_tick_queue_maxsize", 10_000) or 10_000), 1
                )
                self._fallback_tick_queue = queue.Queue(maxsize=maxsize)
            lifecycle_lock = getattr(self, "_tick_worker_lifecycle_lock", None)
            if lifecycle_lock is None:
                lifecycle_lock = threading.RLock()
                self._tick_worker_lifecycle_lock = lifecycle_lock
            with lifecycle_lock:
                thread = getattr(self, "_tick_worker_thread", None)
                if thread is not None and thread.is_alive():
                    return getattr(self, "_tick_worker_state", "RUNNING") == "RUNNING"
                if getattr(self, "_tick_worker_state", "STOPPED") in {
                    "STOPPING",
                    "FAILED_TO_STOP",
                }:
                    return False
                self._tick_worker_stop.clear()
                self._tick_worker_generation = int(
                    getattr(self, "_tick_worker_generation", 0) or 0
                ) + 1
                generation = self._tick_worker_generation
                self._tick_worker_state = "RUNNING"
                self._tick_worker_failed_to_stop = False
                thread = threading.Thread(
                    target=self._tick_worker_loop,
                    name=f"mdm-tick-worker-{generation}",
                    daemon=True,
                )
                self._tick_worker_thread = thread
                thread.start()
            self._logger.info(
                "MDM_FALLBACK_TICK_WORKER_STARTED queue_type=threading generation=%s",
                generation,
                extra={
                    "event": "MDM_FALLBACK_TICK_WORKER_STARTED",
                    "queue_type": "queue.Queue",
                    "generation": generation,
                },
            )
            return True


        def _safe_queue_task_done'''
    ),
    label="replace worker ensure",
    flags=re.S,
)
text = regex_once(
    text,
    r"def _tick_worker_loop_thread_queue\(self: Any\) -> None:\n.*?\n\ndef _stop_fallback_tick_worker",
    dedent(
        '''\
        def _tick_worker_loop_thread_queue(self: Any) -> None:
            """Fallback serial tick worker used only before the asyncio loop is wired."""
            q: queue.Queue[dict[str, Any]] = self._fallback_tick_queue
            current = threading.current_thread()
            try:
                while not self._tick_worker_stop.is_set():
                    try:
                        raw = q.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    try:
                        self._process_queued_tick(raw)
                        self._tick_processed_total += 1
                    except Exception as exc:  # noqa: BLE001
                        self._record_tick_drop(
                            str(raw.get("_mdm_priority_bucket") or "unknown"),
                            "fallback_process_error",
                        )
                        self._logger.error(
                            "MDM_FALLBACK_TICK_WORKER_ERROR error=%r",
                            exc,
                            exc_info=True,
                            extra={
                                "event": "MDM_FALLBACK_TICK_WORKER_ERROR",
                                "error": repr(exc),
                            },
                        )
                    finally:
                        _safe_queue_task_done(q)
            finally:
                lifecycle_lock = getattr(
                    self, "_tick_worker_lifecycle_lock", threading.RLock()
                )
                with lifecycle_lock:
                    if getattr(self, "_tick_worker_thread", None) is current:
                        self._tick_worker_thread = None
                        self._tick_worker_state = "STOPPED"
                        self._tick_worker_failed_to_stop = False


        def _stop_fallback_tick_worker'''
    ),
    label="replace worker loop",
    flags=re.S,
)
text = regex_once(
    text,
    r"def _stop_fallback_tick_worker\(self: Any\) -> None:\n.*?\n\ndef _ensure_candle_flush_task",
    dedent(
        '''\
        def _stop_fallback_tick_worker(self: Any) -> bool:
            """Stop the current owner without making a timed-out worker restartable."""
            lifecycle_lock = getattr(self, "_tick_worker_lifecycle_lock", None)
            if lifecycle_lock is None:
                lifecycle_lock = threading.RLock()
                self._tick_worker_lifecycle_lock = lifecycle_lock
            with lifecycle_lock:
                thread = getattr(self, "_tick_worker_thread", None)
                if thread is None:
                    self._tick_worker_state = "STOPPED"
                    return True
                self._tick_worker_state = "STOPPING"
                self._tick_worker_stop.set()
            if thread is threading.current_thread():
                return False
            timeout = max(
                float(getattr(self, "_tick_worker_stop_timeout_s", 2.0) or 2.0),
                0.05,
            )
            if thread.is_alive():
                thread.join(timeout=timeout)
            with lifecycle_lock:
                if thread.is_alive():
                    self._tick_worker_state = "FAILED_TO_STOP"
                    self._tick_worker_failed_to_stop = True
                    self._logger.critical(
                        "MDM_FALLBACK_TICK_WORKER_STOP_TIMEOUT generation=%s timeout_s=%.3f",
                        getattr(self, "_tick_worker_generation", 0),
                        timeout,
                        extra={
                            "event": "MDM_FALLBACK_TICK_WORKER_STOP_TIMEOUT",
                            "generation": getattr(self, "_tick_worker_generation", 0),
                            "timeout_s": timeout,
                        },
                    )
                    return False
                if getattr(self, "_tick_worker_thread", None) is thread:
                    self._tick_worker_thread = None
                self._tick_worker_state = "STOPPED"
                self._tick_worker_failed_to_stop = False
                return True


        def _ensure_candle_flush_task'''
    ),
    label="replace worker stop",
    flags=re.S,
)
write(path, text)

# ----------------------------------------------------------------------------
# P1: reconcile single-flight plus stale local-generation rejection.
# ---------------------------------------------------------------------------
path = "src/nifty_scalper_bot/execution/position_manager.py"
text = read(path)
text = replace_once(
    text,
    """_BROKER_POSITION_SNAPSHOT_MAX_AGE_MAX_S = 300.0\n\n\ndef _resolve_broker_position_snapshot_max_age_seconds() -> float:\n""",
    """_BROKER_POSITION_SNAPSHOT_MAX_AGE_MAX_S = 300.0\n\n\nclass StaleReconciliationSnapshot(RuntimeError):\n    \"\"\"Broker snapshot fetched before a newer local exposure mutation.\"\"\"\n\n\ndef _resolve_broker_position_snapshot_max_age_seconds() -> float:\n""",
    label="stale reconcile exception",
)
text = replace_once(
    text,
    """        self._reconcile_timer: threading.Timer | None = None\n        self._reconcile_interval_s: float = 60.0\n""",
    """        self._reconcile_timer: threading.Timer | None = None\n        self._reconcile_inflight_lock = threading.Lock()\n        self._reconcile_state_lock = threading.Lock()\n        self._reconcile_request_generation = 0\n        self._last_applied_reconcile_generation = 0\n        self._reconcile_coalesced_requests = 0\n        self._last_reconcile_fetch_latency_s: float | None = None\n        self._last_reconcile_apply_latency_s: float | None = None\n        self._reconcile_interval_s: float = 60.0\n""",
    label="reconcile state initialization",
)
text = replace_once(
    text,
    """    def synchronize_with_broker(\n        self, broker_positions: Sequence[Mapping[str, object]]\n    ) -> None:\n""",
    """    def synchronize_with_broker(\n        self,\n        broker_positions: Sequence[Mapping[str, object]],\n        *,\n        expected_local_generation: int | None = None,\n        reconcile_generation: int | None = None,\n    ) -> None:\n""",
    label="synchronize signature",
)
text = replace_once(
    text,
    """        with self._lock:\n            existing_positions = copy.deepcopy(self._positions)\n""",
    """        with self._lock:\n            if (\n                expected_local_generation is not None\n                and self._local_position_generation != expected_local_generation\n            ):\n                raise StaleReconciliationSnapshot(\n                    \"local position generation changed during broker fetch\"\n                )\n            if (\n                reconcile_generation is not None\n                and reconcile_generation <= self._last_applied_reconcile_generation\n            ):\n                raise StaleReconciliationSnapshot(\n                    \"older reconciliation generation cannot overwrite newer state\"\n                )\n            existing_positions = copy.deepcopy(self._positions)\n""",
    label="stale snapshot gate",
)
text = replace_once(
    text,
    """            self._broker_snapshot_local_generation = self._local_position_generation\n            self._last_broker_position_snapshot_source = snapshot.source\n""",
    """            self._broker_snapshot_local_generation = self._local_position_generation\n            if reconcile_generation is not None:\n                self._last_applied_reconcile_generation = reconcile_generation\n            self._last_broker_position_snapshot_source = snapshot.source\n""",
    label="applied generation record",
)
text = regex_once(
    text,
    r"    def reconcile_now\(self\) -> bool:\n.*?\n    def reconcile_periodic\(",
    dedent(
        ''W\
            def reconcile_now(self) -> bool:
                """Fetch and apply one authoritative broker snapshot, single-flight."""
                if not self._reconcile_inflight_lock.acquire(blocking=False):
                    with self._reconcile_state_lock:
                        self._reconcile_coalesced_requests += 1
                        coalesced = self._reconcile_coalesced_requests
                    self._logger.info(
                        "POSITION_RECONCILE_COALESCED count=%s",
                        coalesced,
                        extra={
                            "event": "POSITION_RECONCILE_COALESCED",
                            "coalesced_requests": coalesced,
                        },
                    )
                    return True

                payload_count = 0
                with self._reconcile_state_lock:
                    self._reconcile_request_generation += 1
                    generation = self._reconcile_request_generation
                with self._lock:
                    expected_local_generation = self._local_position_generation
                fetcher = self._resolve_broker_position_fetcher()
                try:
                    if fetcher is None:
                        self._handle_reconcile_failure(
                            reason=canonical("fetcher_missing"),
                            error=None,
                            payload_count=0,
                            previous_positions=None,
                        )
                        return False
                    fetch_started = time.monotonic()
                    try:
                        response = fetcher()
                        snapshot = decode_position_snapshot(response)
                    except Exception as exc:  # noqa: BLE001
                        reason = canonical(
                            "payload_invalid"
                            if isinstance(exc, PositionSnapshotError)
                          else "fetch_error"
                        )
                        self._logger.warning(
                            "Position reconciliation snapshot failed: %s",
                            exc,
                            extra={
                                "event": "position_reconcile_failed",
                                "reason": reason,
                                "generation": generation,
                            },
                            exc_info=exc,
                        )
                        self._handle_reconcile_failure(
                            reason=reason,
                            error=exc,
                            payload_count=0,
                            previous_positions=None,
                        )
                        return False
                    finally:
                        self._last_reconcile_fetch_latency_s = max(
                            0.0, time.monotonic() - fetch_started
                        )

                    payloads = snapshot.raw_rows()
                    payload_count = len(payloads)
                    apply_started = time.monotonic()
                    try:
                        self.synchronize_with_broker(
                            payloads,
                            expected_local_generation=expected_local_generation,
                            reconcile_generation=generation,
                        )
                    except StaleReconciliationSnapshot as exc:
                        self._logger.warning(
                            "POSITION_RECONCILE_STALE_REJECTED generation=%s local_generation=%s error=%s",
                            generation,
                            expected_local_generation,
                            exc,
                            extra={
                                "event": "POSITION_RECONCILE_STALE_REJECTED",
                                "generation": generation,
                                "expected_local_generation": expected_local_generation,
                                  "error": str(exc),
                            },
                        )
                        self._handle_reconcile_failure(
                            reason=canonical("stale_snapshot"),
                            error=exc,
                            payload_count=payload_count,
                            previous_positions=None,
                        )
                        return False
                    except Exception as exc:  # noqa: BLE001
                        reason = canonical("apply_error")
                        self._logger.warning(
                            "Position reconciliation apply failed: %s",
                            exc,
                            extra={
                                "event": "position_reconcile_failed",
                                "reason": reason,
                                "generation": generation,
                            },
                            exc_info=exc,
                        )
                        self._handle_reconcile_failure(
                            reason=reason,
                            error=exc,
                            payload_count=payload_count,
                            previous_positions=None,
                        )
                        return False
                    finally:
                        self._last_reconcile_apply_latency_s = max(
                            0.0, time.monotonic() - apply_started
                        )

                    self._logger.info(
                        "POSITION_RECONCILE_OK count=%s source=%s generation=%s fetch_ms=%.1f apply_ms=%.1f",
                        payload_count,
                        snapshot.source,
                        generation,
                        self._last_reconcile_fetch_latency_s * 1000.0,
                        self._last_reconcile_apply_latency_s * 1000.0,
                        extra={
                            "event": "position_reconcile_ok",
                            "count": payload_count,
                            "source": snapshot.source,
                            "generation": generation,
                            "fetch_latency_seconds": self._last_reconcile_fetch_latency_s,
                            "apply_latency_seconds": self._last_reconcile_apply_latency_s,
                        },
                    )
                    self._handle_reconcile_success(payload_count)
                    return True
                finally:
                    self._reconcile_inflight_lock.release()

            def reconcile_periodic('''
    ),
    label="replace reconcile_now",
    flags=re.S,
)
write(path, text)

# ---------------------------------------------------------------------------
# P1: one market-aware supervisor with restart backoff and latency SLO state.
# ---------------------------------------------------------------------------
watchdog = dedent('''\
    """Single-owner market-data supervision and latency SLO enforcement."""

    from __future__ import annotations

    from dataclasses import asdict, dataclass
    from enum import Enum
    import logging
    import os
    import threading
    import time
    from typing import Any, Callable, Mapping

    LOGGER = logging.getLogger("nifty_scalper_bot.market_data_supervisor")


    class SupervisorState(str, Enum):
        STARTING = "STARTING"
        HEALTHY = "HEALTHY"
        DEGRADED = "DEGRADDD"
        RECONNECTING = "RECONNECTING"
        MARKET_CLOSED = "MARKET_CLOSED"
        FAILED = "FAILED"
        STOPPING = "STOPPING"
        STOPPED = "STOPPED"

    @dataclass(frozen=True, slots=True)
    class SupervisorSnapshot:
        state: str
        latency_degraded: bool
        event_loop_lag_ms: float
        tick_p99_ms: float
        queue_age_ms: float
        queue_utilization: float
        open_position_quote_age_s: float
        consecutive_failures: int
        restart_count: int
        fatal_stale_samples: int

        def to_dict(self) -> dict[str, object]:
            return asdict(self)


    def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
        try:
            value = float(os.getenv(name, str(default)) or default)
        except (TypeError, ValueError):
            value = default
        return max(value, minimum)


    def _int_env(name: str, default: int, minimum: int = 1) -> int:
        try:
            value = int(os.getenv(name, str(default)) or default)
        except (TypeError, ValueError):
            value = default
        return max(value, minimum)


    def _default_market_open() -> bool:
        try:
            from nifty_scalper_bot.utils.market_hours import get_runtime_market_mode

            token = str(get_runtime_market_mode()).upper()
            return "OPEN" in token and "POST" not in token
        except Exception:
            return True


    def _age_seconds(raw: object, now: float) -> float | None:
        try:
            value = float(raw or 0.0)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return max(0.0, now - value)


    class MarketDataSupervisor:
        """One deterministic owner for restart policy and latency degradation."""

        def __init__(
            self,
            market_data_manager: Any,
            *,
            market_open_getter: Callable[[], bool] = _default_market_open,
            exit_process: Callable[[int], Any] = os._exit,
            interval_s: float | None = None,
         ) -> None:
            self.mdm = market_data_manager
            self.market_open_getter = market_open_getter
            self.exit_process = exit_process
            self.interval_s = interval_s or _float_env(
                "MAR
âÌz{mÆÈ‹j◊ùvÁ≠¢»≥