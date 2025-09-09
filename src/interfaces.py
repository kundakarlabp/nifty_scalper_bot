"""Core component interfaces used across the application."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Protocol

import pandas as pd

from src.data.types import HistResult


class DataSource(Protocol):
    """Minimal interface expected from data providers."""

    def connect(self) -> None: ...

    def fetch_ohlc(
        self, token: int, start: datetime, end: datetime, timeframe: str
    ) -> HistResult: ...

    def get_last_price(self, symbol_or_token: Any) -> Optional[float]: ...

    def api_health(self) -> Dict[str, Dict[str, object]]: ...


class Strategy(Protocol):
    """Strategy plug-in contract."""

    def name(self) -> str: ...

    def evaluate(self, *args: Any, **kwargs: Any) -> Any: ...


class Sizer(Protocol):
    """Position sizing component."""

    def size(self, *args: Any, **kwargs: Any) -> Any: ...


class Executor(Protocol):
    """Order execution engine."""

    def place_order(self, *args: Any, **kwargs: Any) -> Any: ...

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any: ...


class Notifier(Protocol):
    """User notification component."""

    def send(self, text: str) -> None: ...
