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
        '''\
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

