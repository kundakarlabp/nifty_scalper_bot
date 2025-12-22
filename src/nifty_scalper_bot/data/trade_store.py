"""
Trade Persistence Layer.
Saves trade intent to disk to prevent duplicate orders on bot restarts.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Any
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
    status: str = "PENDING"  # PENDING, SUBMITTED, FILLED, REJECTED
    broker_order_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class TradeStore:
    def __init__(self, filepath="data/trades.json"):
        self.filepath = filepath
        self._trades: Dict[str, TradeIntent] = {}
        self._ensure_dir()
        self._load()

    def _ensure_dir(self):
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _load(self):
        """Load trades from disk safely."""
        if not os.path.exists(self.filepath):
            return

        try:
            with open(self.filepath, 'r') as f:
                content = f.read().strip()
                if not content:
                    self._trades = {}
                    return
                
                data = json.loads(content)

                # ✅ FIX 1: Handle List (Legacy/Corrupted)
                if isinstance(data, list):
                    LOGGER.warning("Trade store found as list. Resetting to empty dict to prevent crash.")
                    self._trades = {} 
                    return

                # ✅ FIX 2: Handle Dict (Normal)
                if isinstance(data, dict):
                    self._trades = {k: TradeIntent.from_dict(v) for k, v in data.items()}
                else:
                    LOGGER.warning(f"Unknown trade store format: {type(data)}. Resetting.")
                    self._trades = {}

            LOGGER.info(f"Loaded {len(self._trades)} trades from store.")

        except Exception as e:
            LOGGER.error(f"Failed to load trade store: {e}")
            # Fallback to empty to ensure bot starts even if file is bad
            self._trades = {}

    def save(self):
        """Atomic save to disk."""
        try:
            temp_path = self.filepath + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump({k: asdict(v) for k, v in self._trades.items()}, f, indent=2)
            os.replace(temp_path, self.filepath)
        except Exception as e:
            LOGGER.error(f"Failed to save trade store: {e}")

    def add_trade(self, trade: TradeIntent) -> bool:
        """Register a new trade intent. Returns False if already exists."""
        if self.exists_by_signal(trade.signal_id):
            return False
        self._trades[trade.trade_id] = trade
        self.save()
        return True

    def exists_by_signal(self, signal_id: str) -> bool:
        """Check if we already acted on this signal ID."""
        # Check if any trade has this signal_id
        for t in self._trades.values():
            if t.signal_id == signal_id:
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
