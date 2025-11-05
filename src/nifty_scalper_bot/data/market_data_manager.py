"""Central market data manager responsible for tick fan-out and broker cache."""
from __future__ import annotations
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import os
from random import uniform
import threading
import time
from typing import Any, Callable, Deque, Iterable, Mapping, Sequence, cast
from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.data.resolver import InstrumentResolver
from nifty_scalper_bot.data.websocket.manager import ConnectionState, WebSocketManager
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.env import get_str
from nifty_scalper_bot.utils.logging import get_logger, get_tracer_logger
from nifty_scalper_bot.utils.metrics import Counter
TickCallback = Callable[[dict[str, Any]], None]
_EXPIRY_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d-%b-%Y",
    "%d-%b-%Y %H:%M",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S.%f",
    "%d %b %Y",
    "%d %b %Y %H:%M",
    "%d %b %Y %H:%M:%S",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
)
_COMPACT_EXPIRY_FORMATS: tuple[str, ...] = (
    "%d%b%Y", "%d%b%y"
)
_logger = get_tracer_logger(__name__)
@dataclass(slots=True)
class _OHLCBar:
    """Normalized one-minute OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
class _OHLCBuilder:
    """Aggregate ticks into fixed one-minute OHLC bars."""
    def __init__(self, *, maxlen: int = 500) -> None:
        self._bars: dict[str, Deque[_OHLCBar]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._last_cumulative_volume: dict[str, float] = {}
        self._lock = threading.RLock()

# (... the rest of the file ...)

class MarketDataManager:
    def __init__(self, resolver: InstrumentResolver | None = None, ...):
        self._resolver = resolver
        # Warm up the resolver to ensure instrument cache is populated
        if self._resolver is not None:
            _logger.info("Warming up InstrumentResolver...")
            self._resolver.warm()
            _logger.info("InstrumentResolver warmup complete")
        # ...rest of the init code...
