"""Sampled logging utilities for regime and other diagnostic events.

Goals:
- Keep behavior simple and predictable.
- Stop repeated near-identical messages flooding logs.
- Emit when state changes or at most once per `period_s`.
- Additionally dedupe identical events for `event_dedupe_s` seconds.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Dict, Optional, Protocol, Tuple

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class _SupportsLogMethod(Protocol):
    def __call__(self, msg: str, *, extra: Dict[str, Any] | None = None) -> None:
        """Log a message."""


@dataclass(slots=True)
class _State:
    last_regime: Optional[str] = None
    last_multiplier: Optional[float] = None
    last_emit_t: float = 0.0


class RegimeSampler:
    """Simple sampler that avoids log spam while preserving material-change emits.

    Parameters
    ----------
    period_s:
        Wall-clock minimum seconds between *any* real emission from this instance.
    tol:
        Absolute tolerance for multiplier change detection.
    event_dedupe_s:
        Additional per-event dedupe window (identical event key will be suppressed
        if emitted less than event_dedupe_s seconds ago). Set to 0 to disable.
    """

    def __init__(self, *, period_s: float = 15.0, tol: float = 1e-6, event_dedupe_s: float = 5.0) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be > 0")
        if event_dedupe_s < 0:
            raise ValueError("event_dedupe_s must be >= 0")

        self._period_s = float(period_s)
        self._tol = float(tol)
        self._event_dedupe_s = float(event_dedupe_s)

        self._st = _State()
        self._lock = threading.Lock()

        # Per-instance last emission time (global wall-clock guard)
        self._last_real_log_time: Optional[float] = None

        # Per-event last emit times: key -> monotonic timestamp
        # Key is derived from (event, regime, rounded_multiplier)
        self._event_last_emit: Dict[str, float] = {}

        # Counters for visibility (not used to change behavior)
        self._suppressed_count: int = 0
        self._emitted_count: int = 0

        LOGGER.debug(
            "RegimeSampler initialized",
            extra={"event": "regime_sampler_init", "period_s": self._period_s, "tol": self._tol, "event_dedupe_s": self._event_dedupe_s},
        )

    # --- public helpers ----------------------------------------------------
    def get_and_reset_counters(self) -> Tuple[int, int]:
        """Return (suppressed, emitted) counts and reset them atomically."""
        with self._lock:
            suppressed = self._suppressed_count
            emitted = self._emitted_count
            self._suppressed_count = 0
            self._emitted_count = 0
        return suppressed, emitted

    # --- internal helpers --------------------------------------------------
    def _changed(self, regime: Optional[str], multiplier: Optional[float]) -> bool:
        try:
            if regime != self._st.last_regime:
                return True
            a = self._st.last_multiplier
            b = multiplier
            if a is None or b is None:
                return True
            return not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=self._tol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failure in RegimeSampler._changed", extra={"event": "regime_sampler_changed_error", "error": str(exc)})
            # Fail-open: treat as change
            return True

    def _event_key(self, event: str, regime: Optional[str], multiplier: Optional[float]) -> str:
        """Canonical string key for per-event dedupe map."""
        mult_key = "None" if multiplier is None else f"{float(round(multiplier, 6))}"
        return f"{event}|{regime or 'UNKNOWN'}|{mult_key}"

    # --- main entry -------------------------------------------------------
    def maybe_log(
        self,
        logger: Any,
        *,
        event: str = "regime_multiplier_applied",
        regime: Optional[str],
        multiplier: Optional[float | int],
        extra: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """Emit a structured log only on material change or when allowed by throttles.

        Behavior summary:
        - If regime/multiplier changed materially -> candidate to emit.
        - Otherwise, allow emission only if (now - last_emit) >= period_s.
        - Additionally suppress identical (event, regime, multiplier) emissions for event_dedupe_s.
        - Maintain simple counters for suppressed/emitted events.
        """
        try:
            now = monotonic()

            # Quick pre-check: global instance wall-clock guard
            last_real = self._last_real_log_time
            if last_real is not None and (now - last_real) < self._period_s:
                # We'll still consult per-event dedupe for better accounting, but
                # for performance bail early to avoid lock churn.
                with self._lock:
                    self._suppressed_count += 1
                return

            target_multiplier = None if multiplier is None else float(multiplier)

            # Decide emit based on change detection or period expiry
            with self._lock:
                must_emit = False
                if self._changed(regime, target_multiplier):
                    must_emit = True
                elif now - self._st.last_emit_t >= self._period_s:
                    must_emit = True

                if not must_emit:
                    self._suppressed_count += 1
                    return

                # Per-event dedupe check (to avoid many identical messages)
                key = self._event_key(event, regime, target_multiplier)
                last_event_t = self._event_last_emit.get(key)
                if last_event_t is not None and (now - last_event_t) < self._event_dedupe_s:
                    # Suppress identical event within dedupe window
                    self._suppressed_count += 1
                    return

                # Update state and emit
                self._st.last_regime = regime
                self._st.last_multiplier = target_multiplier
                self._st.last_emit_t = now
                self._last_real_log_time = now
                self._event_last_emit[key] = now
                self._emitted_count += 1

            # build payload and log outside lock
            payload: Dict[str, Any] = {
                "event": event,
                "regime": regime or "UNKNOWN",
                "multiplier": None if target_multiplier is None else round(float(target_multiplier), 6),
            }
            if extra:
                payload.update(extra)

            log_fn: _SupportsLogMethod | None = getattr(logger, level, None)
            if log_fn is None:
                log_fn = getattr(logger, "info")
            log_fn(f"Condition met: {event}", extra=payload)

        except Exception as exc:  # noqa: BLE001
            # Ensure any failure here is visible but doesn't crash the caller
            LOGGER.exception("Failure in RegimeSampler.maybe_log", extra={"event": "regime_sampler_maybe_log_error", "error": str(exc)})


__all__ = ["RegimeSampler"]
