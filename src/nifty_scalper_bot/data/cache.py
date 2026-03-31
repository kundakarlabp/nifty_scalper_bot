"""TTL cache with background cleanup thread."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Tuple


class TTLCache:
    """Simple TTL cache supporting background cleanup."""

    def __init__(self, ttl_sec: float) -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")
        self._ttl = ttl_sec
        self._data: Dict[Any, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()

    def set(self, key: Any, value: Any) -> None:
        expires = time.monotonic() + self._ttl
        with self._lock:
            self._data[key] = (value, expires)

    def get(self, key: Any) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires = entry
            if expires < time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(self._ttl):
            now = time.monotonic()
            with self._lock:
                keys_to_remove = [
                    key for key, (_, expires) in self._data.items() if expires < now
                ]
                for key in keys_to_remove:
                    self._data.pop(key, None)


__all__ = ["TTLCache"]
