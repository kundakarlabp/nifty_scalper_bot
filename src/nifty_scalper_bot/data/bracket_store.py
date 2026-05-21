import sqlite3
import json
import time
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Optional
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.serialization import to_json_safe

LOGGER = get_logger(__name__)

class BracketStore:
    """
    Production-Grade Persistence using SQLite.
    ACID compliant: Prevents data corruption during crashes.
    """
    def __init__(self, db_path: str = "data/bot_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Creates the table if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS virtual_brackets (
                        order_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        created_at REAL,
                        updated_at REAL,
                        state_json TEXT NOT NULL
                    )
                """)
                # Create an index/archive table for history if needed later
                conn.execute("VACUUM") # Optimize size on startup
        except Exception as e:
            LOGGER.critical(f"❌ Failed to initialize SQLite DB: {e}")

    def save_bracket(self, bracket_data: Dict[str, Any]) -> None:
        """
        Saves or Updates a SINGLE bracket. Efficient and Safe.
        """
        try:
            order = bracket_data if isinstance(bracket_data, dict) else {}
            order_id = order.get("order_id") if order else None
            if not order_id:
                order_id = f"VIRTUAL-{uuid4().hex}"
                bracket_data["order_id"] = order_id
            symbol = str(bracket_data.get("symbol") or "UNKNOWN")
            # We serialize the full data object to JSON to store in one flexible column
            json_dump = json.dumps(to_json_safe(bracket_data), ensure_ascii=False, default=str)
            now = time.time()

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO virtual_brackets (order_id, symbol, created_at, updated_at, state_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(order_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        state_json = excluded.state_json
                """, (order_id, symbol, now, now, json_dump))
        except Exception as e:
            LOGGER.error(f"❌ DB Write Error for {bracket_data.get('symbol')}: {e}")

    def delete_bracket(self, order_id: str) -> None:
        """Removes a bracket when the trade is closed."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM virtual_brackets WHERE order_id = ?", (order_id,))
        except Exception as e:
            LOGGER.error(f"❌ DB Delete Error for {order_id}: {e}")

    def load_all_brackets(self) -> List[Dict[str, Any]]:
        """Loads all active brackets on startup."""
        brackets = []
        if not self.db_path.exists():
            return brackets

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT state_json FROM virtual_brackets")
                rows = cursor.fetchall()
                
                for row in rows:
                    try:
                        data = json.loads(row[0])
                        brackets.append(data)
                    except json.JSONDecodeError:
                        continue
            
            LOGGER.info(f"✅ Loaded {len(brackets)} brackets from SQLite.")
            return brackets
        except Exception as e:
            LOGGER.error(f"❌ DB Read Error: {e}")
            return []
