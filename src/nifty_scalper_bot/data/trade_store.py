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
    """Store for trade intents with Railway-compatible paths.
    
    ✅ PRODUCTION FIX: Uses DATA_DIR env var with /tmp fallback.
    """
    
    def __init__(self, filepath=None):
        """Initialize TradeStore with DATA_DIR support for Railway.
        
        ✅ PRODUCTION FIX: Uses DATA_DIR environment variable.
        """
        import os
        
        if filepath is None:
            # ✅ FIX: Use DATA_DIR environment variable
            data_dir = os.getenv("DATA_DIR", "data")
            filepath = os.path.join(data_dir, "trades.json")
        
        self.filepath = filepath
        self._trades: Dict[str, TradeIntent] = {}
        self._ensure_dir()
        self._load()

    def _ensure_dir(self):
        """Create directory with /tmp fallback on permission error.
        
        ✅ PRODUCTION FIX: Falls back to /tmp if main directory is read-only.
        """
        import os
        
        directory = os.path.dirname(self.filepath)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
                # Test write permission
                test_file = os.path.join(directory, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                LOGGER.debug(f"✅ TradeStore directory ready: {directory}")
            except (PermissionError, OSError) as e:
                # ✅ FIX: Fallback to /tmp
                fallback_dir = "/tmp/nifty_scalper_data"
                os.makedirs(fallback_dir, exist_ok=True)
                old_path = self.filepath
                self.filepath = os.path.join(fallback_dir, "trades.json")
                LOGGER.warning(f"⚠️ Permission denied on {old_path}, using fallback: {self.filepath}")

    def _load(self):
        """Load trades from disk safely."""
        if not os.path.exists(self.filepath):
            # Also check fallback location
            fallback = "/tmp/nifty_scalper_data/trades.json"
            if os.path.exists(fallback):
                self.filepath = fallback
                LOGGER.info(f"📂 Loading trades from fallback: {fallback}")
            else:
                return

        try:
            with open(self.filepath, 'r') as f:
                content = f.read().strip()
                if not content:
                    self._trades = {}
                    return
                
                data = json.loads(content)

                # Handle List (Legacy/Corrupted)
                if isinstance(data, list):
                    LOGGER.warning("Trade store found as list. Resetting to empty dict.")
                    self._trades = {} 
                    return

                # Handle Dict (Normal)
                if isinstance(data, dict):
                    self._trades = {k: TradeIntent.from_dict(v) for k, v in data.items()}
                else:
                    LOGGER.warning(f"Unknown trade store format: {type(data)}. Resetting.")
                    self._trades = {}

            LOGGER.info(f"✅ Loaded {len(self._trades)} trades from {self.filepath}")

        except Exception as e:
            LOGGER.error(f"Failed to load trade store: {e}")
            self._trades = {}

    def save(self):
        """Atomic save to disk with Enum handling.
        
        ✅ PRODUCTION FIX: Added unique temp file and Enum serialization.
        """
        import uuid
        from enum import Enum
        from datetime import datetime, date
        from decimal import Decimal
        
        def _sanitize(obj):
            """Recursively convert non-JSON-serializable types."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_sanitize(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value if hasattr(obj, 'value') else obj.name
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj
        
        try:
            # Ensure directory exists
            directory = os.path.dirname(self.filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # ✅ FIX: Use unique temp file to prevent race conditions
            temp_path = f"{self.filepath}.tmp.{uuid.uuid4().hex}"
            
            # Sanitize data before serialization
            data = {k: _sanitize(asdict(v)) for k, v in self._trades.items()}
            
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(temp_path, self.filepath)
            LOGGER.debug(f"✅ Trades saved to {self.filepath}")
            
        except PermissionError as e:
            # ✅ FIX: Try fallback location
            fallback_path = "/tmp/nifty_scalper_data/trades.json"
            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            
            data = {k: _sanitize(asdict(v)) for k, v in self._trades.items()}
            temp_path = f"{fallback_path}.tmp.{uuid.uuid4().hex}"
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(temp_path, fallback_path)
            
            self.filepath = fallback_path
            LOGGER.warning(f"⚠️ Saved trades to fallback: {fallback_path}")
            
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
