# src/utils/logging_tools.py
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple


class InMemoryLogHandler(logging.Handler):
    """
    Ring-buffer log handler so we can fetch recent logs via Telegram (/logs).
    Stores (ts, levelno, formatted_message). Thread-safe.
    """
    def __init__(self, capacity: int = 4000) -> None:
        super().__init__()
        self.capacity = int(os.environ.get("LOG_BUFFER_CAPACITY", capacity))
        self._buf: Deque[Tuple[float, int, str]] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = getattr(record, "message", str(record))
        with self._lock:
            self._buf.append((time.time(), int(record.levelno), msg))

    def tail(self, n: int = 60, min_level: Optional[int] = None) -> List[str]:
        n = max(1, int(n))
        with self._lock:
            items = list(self._buf)
        if min_level is not None:
            items = [x for x in items if x[1] >= int(min_level)]
        return [m for _, _, m in items[-n:]]


# singleton instance wired by main.py
log_buffer_handler = InMemoryLogHandler()
def get_recent_logs(n: int = 60, min_level: Optional[int] = None) -> str:
    """Return the last N log lines as one string."""
    return "\n".join(log_buffer_handler.tail(n=n, min_level=min_level))
