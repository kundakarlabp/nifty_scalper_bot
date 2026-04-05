"""Async SQLite journal database."""

from __future__ import annotations

import aiosqlite
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    underlying TEXT NOT NULL,
    option_type TEXT NOT NULL,
    strike REAL NOT NULL,
    expiry TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity INTEGER NOT NULL,
    gross_pnl REAL,
    brokerage REAL,
    net_pnl REAL,
    sl_price REAL,
    tp1_price REAL,
    tp2_price REAL,
    exit_reason TEXT,
    regime_at_entry TEXT,
    confidence_at_entry REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    duration_minutes REAL,
    r_multiple_achieved REAL,
    is_winner INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
"""


class AsyncDB:
    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def execute(self, sql: str, params: tuple = ()) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(sql, params)
            await db.commit()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        rows = await self.fetchall(sql, params)
        return rows[0] if rows else None
