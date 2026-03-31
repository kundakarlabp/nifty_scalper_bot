"""Sampled logging utilities for regime events."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class _SupportsLogMethod(Protocol):
    """Protocol for logger methods used by RegimeSampler."""

    def __call__(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log a message."""


@dataclass(slots=True)
class _State:
    """Track the latest emitted regime and multiplier."""

    last_regime: str | None = None
    last_multiplier: float | None = None
    last_emit_t: float = 0.0


class RegimeSampler:
    """Throttle regime multiplier logs while emitting on material change."""

    def __init__(self, *, period_s: float = 15.0, tol: float = 1e-6) -> None:
        """Initialize a sampler that emits at most once per period.

        Args:
            period_s: Minimum seconds between repeated logs.
            tol: Absolute tolerance before detecting multiplier drift.

        Returns:
            None.

        Raises:
            ValueError: If ``period_s`` is non-positive.
        """

        LOGGER.debug(
            "Entered RegimeSampler.__init__",
            extra={"event": "regime_sampler_init", "period_s": period_s, "tol": tol},
        )
        if period_s <= 0:
            raise ValueError("period_s must be > 0")
        self._period_s = float(period_s)
        self._tol = float(tol)
        self._st = _State()
        self._lock = threading.Lock()

    def _changed(self, regime: str | None, multiplier: float | None) -> bool:
        """Return True when the new regime or multiplier differs.

        Args:
            regime: Candidate regime label.
            multiplier: Candidate multiplier value.

        Returns:
            bool: ``True`` when a material change is detected.

        Raises:
            None.
        """

        try:
            if regime != self._st.last_regime:
                return True
            a = self._st.last_multiplier
            b = multiplier
            if a is None or b is None:
                return True
            return not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=self._tol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in RegimeSampler._changed: %s",
                exc,
                extra={"event": "regime_sampler_changed_error"},
                exc_info=exc,
            )
            return True

    def maybe_log(
        self,
        logger: Any,
        *,
        regime: str | None,
        multiplier: float | int | None,
        extra: dict[str, Any] | None = None,
        level: str = "info",
    ) -> None:
        """Emit sampled regime logs when state changes or interval elapses.

        Args:
            logger: Logger exposing level-named methods (e.g., ``info``).
            regime: Regime label or ``None`` when undefined.
            multiplier: Applied multiplier for the current tick.
            extra: Optional metadata to merge into the log payload.
            level: Logger level name to call when emitting.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            now = monotonic()
            with self._lock:
                must_emit = False

                target_multiplier = (
                    float(multiplier) if multiplier is not None else None
                )
                if self._changed(regime, target_multiplier):
                    must_emit = True
                elif now - self._st.last_emit_t >= self._period_s:
                    must_emit = True

                if not must_emit:
                    return

                self._st.last_regime = regime
                self._st.last_multiplier = target_multiplier
                self._st.last_emit_t = now

            payload: dict[str, Any] = {
                "event": "regime_multiplier_applied",
                "regime": regime or "UNKNOWN",
                "multiplier": (
                    None
                    if target_multiplier is None
                    else round(float(target_multiplier), 6)
                ),
            }
            if extra:
                payload.update(extra)

            log_fn: _SupportsLogMethod | None = getattr(logger, level, None)
            if log_fn is None:
                log_fn = getattr(logger, "info")
            log_fn("Condition met: regime_multiplier_applied", extra=payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in RegimeSampler.maybe_log: %s",
                exc,
                extra={"event": "regime_sampler_maybe_log_error"},
                exc_info=exc,
            )


__all__ = ["RegimeSampler"]
