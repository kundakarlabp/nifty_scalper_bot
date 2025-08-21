from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, List, Optional


class InMemoryLogHandler(logging.Handler):
    """
    Thread-safe ring buffer for recent log lines.
    """
    def __init__(self, capacity: int = 5000, fmt: Optional[str] = None) -> None:
        super().__init__()
        self.capacity = max(200, int(capacity))
        self._buf: Deque[str] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter(fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            line = f"[logfmt-error] {record.getMessage()}"
        with self._lock:
            self._buf.append(line)

    def get_last(self, n: int = 100) -> List[str]:
        n = max(1, min(int(n), self.capacity))
        with self._lock:
            return list(list(self._buf)[-n:])