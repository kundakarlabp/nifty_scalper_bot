"""Simple log emission throttling helper."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class LogGate:
    """Gatekeeper used to throttle repetitive log messages.

    The gate tracks the last emission timestamp for each key. Calls to
    :meth:`should_emit` return ``True`` when a message is allowed to be logged
    based on the configured interval or when ``force`` is provided.
    """

    interval_s: float = 1.0
    _last_emit: dict[str, float] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        try:
            self.interval_s = float(self.interval_s)
        except Exception:  # pragma: no cover - defensive
            self.interval_s = 1.0
        if self.interval_s < 0.0:
            self.interval_s = 0.0

    def should_emit(self, key: str, *, force: bool = False, now: float | None = None) -> bool:
        """Return ``True`` when the log identified by ``key`` should emit."""

        timestamp = now if now is not None else time.monotonic()
        if force or self.interval_s == 0.0:
            with self._lock:
                self._last_emit[key] = timestamp
            return True

        with self._lock:
            last = self._last_emit.get(key)
            if last is None or timestamp - last >= self.interval_s:
                self._last_emit[key] = timestamp
                return True
        return False

    def reset(self, key: str | None = None) -> None:
        """Clear cached emission timestamps.

        Parameters
        ----------
        key:
            Specific key to reset. When ``None`` all cached timestamps are
            cleared.
        """

        with self._lock:
            if key is None:
                self._last_emit.clear()
            else:
                self._last_emit.pop(key, None)


__all__ = ["LogGate"]

