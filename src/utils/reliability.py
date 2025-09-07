from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class RateLimiter:
    """Simple N-per-minute rate limiter."""

    max_per_minute: int
    _ts: Deque[float] = field(default_factory=deque)

    def allow(self) -> bool:
        """Return ``True`` if another action is permitted."""
        now = time.time()
        window_start = now - 60
        while self._ts and self._ts[0] <= window_start:
            self._ts.popleft()
        if len(self._ts) >= self.max_per_minute:
            return False
        self._ts.append(now)
        return True
