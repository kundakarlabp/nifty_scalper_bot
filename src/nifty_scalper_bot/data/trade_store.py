"""Trade persistence layer."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TradeIntent:
    trade_id: str
    symbol: str
    signal_id: str
    strategy: str
    side: str
    qty: int
    timestamp: float
    status: str = 'PENDING'
    broker_order_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TradeIntent':
        """Build dataclass from dictionary. Args: data. Returns: TradeIntent. Raises: KeyError."""
        return cls(**data)


class TradeStore:
    """Store trade intents on disk. Args: filepath. Returns: None. Raises: OSError."""

    def __init__(self, filepath: str | None = None):
        self.filepath = str(Path(filepath)) if filepath else str(get_data_dir() / 'trades.json')
        self._trades: Dict[str, TradeIntent] = {}
        self._ensure_dir()
        self._load()

    def _ensure_dir(self) -> None:
        """Ensure parent directory exists. Args: none. Returns: None. Raises: OSError."""
        path = Path(self.filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load persisted trades. Args: none. Returns: None. Raises: None."""
        path = Path(self.filepath)
        if not path.exists():
            return
        try:
            content = path.read_text(encoding='utf-8').strip()
            if not content:
                self._trades = {}
                return
            data = json.loads(content)
            if isinstance(data, dict):
                self._trades = {k: TradeIntent.from_dict(v) for k, v in data.items()}
            else:
                self._trades = {}
        except Exception as e:
            LOGGER.error('Failure in TradeStore._load: %s', e)
            self._trades = {}

    def save(self) -> None:
        """Atomically persist trades. Args: none. Returns: None. Raises: None."""

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(item) for item in obj]
            if isinstance(obj, Enum):
                return obj.value if hasattr(obj, 'value') else obj.name
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            return obj

        try:
            path = Path(self.filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = path.with_suffix('.tmp')
            payload = {k: _sanitize(asdict(v)) for k, v in self._trades.items()}
            with tmp_file.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_file.replace(path)
        except Exception as e:
            LOGGER.error('Failure in TradeStore.save: %s', e)

    def add_trade(self, trade: TradeIntent) -> bool:
        if self.exists_by_signal(trade.signal_id):
            return False
        self._trades[trade.trade_id] = trade
        self.save()
        return True

    def exists_by_signal(self, signal_id: str) -> bool:
        for trade in self._trades.values():
            if trade.signal_id == signal_id:
                return True
        return False

    def get_trade_by_id(self, trade_id: str) -> Optional[TradeIntent]:
        return self._trades.get(trade_id)

    def update_status(self, trade_id: str, status: str, broker_id: str = None):
        if trade_id in self._trades:
            self._trades[trade_id].status = status
            if broker_id:
                self._trades[trade_id].broker_order_id = broker_id
            self.save()
