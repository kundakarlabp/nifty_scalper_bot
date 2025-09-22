from __future__ import annotations

"""SQLite-backed append-only journal utilities."""

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from src.utils.jsonsafe import json_safe

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  trade_id TEXT NOT NULL,
  leg_id TEXT NOT NULL,
  etype TEXT NOT NULL,
  broker_order_id TEXT,
  idempotency_key TEXT,
  payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_trade ON events(trade_id);
CREATE INDEX IF NOT EXISTS idx_events_leg ON events(leg_id);
CREATE INDEX IF NOT EXISTS idx_events_boid ON events(broker_order_id);

CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_entry TEXT, ts_exit TEXT,
  trade_id TEXT, side TEXT, symbol TEXT, qty INTEGER,
  entry REAL, exit REAL, exit_reason TEXT,
  R REAL, pnl_R REAL, pnl_rupees REAL
);
CREATE INDEX IF NOT EXISTS idx_trades_trade ON trades(trade_id);

CREATE TABLE IF NOT EXISTS idempotency(
  idempotency_key TEXT PRIMARY KEY,
  leg_id TEXT NOT NULL,
  ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints(
  ts TEXT PRIMARY KEY,
  payload TEXT
);
"""


@dataclass
class Journal:
    """Lightweight SQLite journal for order events and trades."""

    path: str
    _conn: sqlite3.Connection
    _lock: threading.Lock

    @classmethod
    def open(cls, path: str) -> "Journal":
        """Open a journal at ``path``, creating tables if needed."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        with conn:
            for stmt in filter(None, DDL.split(";")):
                s = stmt.strip()
                if s:
                    conn.execute(s)
        return cls(path=path, _conn=conn, _lock=threading.Lock())

    # ---------------- internal helpers ----------------
    def _exec(self, sql: str, args: Iterable[Any] = ()):
        with self._lock:
            return self._conn.execute(sql, tuple(args))

    # ---------------- events ----------------
    def append_event(
        self,
        *,
        ts: str,
        trade_id: str,
        leg_id: str,
        etype: str,
        broker_order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an event row and record idempotency mapping if provided."""
        self._exec(
            "INSERT INTO events(ts,trade_id,leg_id,etype,broker_order_id,idempotency_key,payload) VALUES(?,?,?,?,?,?,?)",
            (
                ts,
                trade_id,
                leg_id,
                etype,
                broker_order_id,
                idempotency_key,
                json.dumps(payload or {}),
            ),
        )
        if idempotency_key:
            self._exec(
                "INSERT OR REPLACE INTO idempotency(idempotency_key,leg_id,ts) VALUES(?,?,?)",
                (idempotency_key, leg_id, ts),
            )

    # ---------------- trades ----------------
    def append_trade(self, trade: Dict[str, Any]) -> None:
        """Append a closed trade record."""
        self._exec(
            "INSERT INTO trades(ts_entry,ts_exit,trade_id,side,symbol,qty,entry,exit,exit_reason,R,pnl_R,pnl_rupees) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade.get("ts_entry"),
                trade.get("ts_exit"),
                trade.get("trade_id"),
                trade.get("side"),
                trade.get("symbol"),
                int(trade.get("qty", 0)),
                float(trade.get("entry", 0.0)),
                float(trade.get("exit", 0.0)),
                trade.get("exit_reason"),
                float(trade.get("R", 0.0)),
                float(trade.get("pnl_R", 0.0)),
                float(trade.get("pnl_rupees", 0.0)),
            ),
        )

    def last_trades(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the latest ``n`` closed trades."""
        cur = self._exec(
            "SELECT ts_entry,ts_exit,trade_id,side,symbol,qty,entry,exit,exit_reason,R,pnl_R,pnl_rupees "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        keys = [
            "ts_entry",
            "ts_exit",
            "trade_id",
            "side",
            "symbol",
            "qty",
            "entry",
            "exit",
            "exit_reason",
            "R",
            "pnl_R",
            "pnl_rupees",
        ]
        return [dict(zip(keys, r)) for r in rows]

    # ---------------- idempotency ----------------
    def get_idemp_leg(self, key: str) -> Optional[str]:
        """Return leg_id associated with ``key`` if known."""
        cur = self._exec(
            "SELECT leg_id FROM idempotency WHERE idempotency_key=?", (key,)
        )
        r = cur.fetchone()
        return r[0] if r else None

    # ---------------- checkpoints ----------------
    def save_checkpoint(self, payload: Dict[str, Any]) -> None:
        """Persist a lightweight checkpoint snapshot."""
        snap = dict(payload)
        ts = snap.get("ts") or datetime.utcnow().isoformat()
        if not isinstance(ts, str):
            try:
                ts = ts.isoformat()
            except Exception:  # pragma: no cover - fallback path
                ts = str(ts)
        snap["ts"] = ts
        payload_json = json.dumps(snap, default=json_safe, ensure_ascii=False)
        self._exec(
            "INSERT INTO checkpoints(ts,payload) VALUES(?,?)", (ts, payload_json)
        )

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Return latest checkpoint payload if present."""
        cur = self._exec("SELECT payload FROM checkpoints ORDER BY ts DESC LIMIT 1")
        r = cur.fetchone()
        return json.loads(r[0]) if r else None

    # ---------------- rehydrate ----------------
    def rehydrate_open_legs(self) -> List[Dict[str, Any]]:
        """Return non-terminal legs with their latest known state."""
        cur = self._exec(
            """
           SELECT e1.trade_id, e1.leg_id, e1.etype, e1.broker_order_id, e1.idempotency_key, e1.payload
           FROM events e1
           JOIN (SELECT leg_id, MAX(id) AS max_id FROM events GROUP BY leg_id) t
             ON e1.leg_id=t.leg_id AND e1.id=t.max_id
        """
        )
        open_legs: List[Dict[str, Any]] = []
        for trade_id, leg_id, etype, boid, ikey, payload_json in cur.fetchall():
            if etype in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                continue
            payload = json.loads(payload_json or "{}")
            if (
                not payload.get("side")
                or not payload.get("symbol")
                or not payload.get("qty")
            ):
                cur2 = self._exec(
                    "SELECT payload FROM events WHERE leg_id=? AND etype='NEW' ORDER BY id ASC LIMIT 1",
                    (leg_id,),
                )
                row2 = cur2.fetchone()
                base = json.loads(row2[0]) if row2 else {}
                for key in ["side", "symbol", "qty", "limit_price"]:
                    if key not in payload or payload.get(key) is None:
                        if base.get(key) is not None:
                            payload[key] = base.get(key)
            if not ikey:
                cur3 = self._exec(
                    "SELECT idempotency_key FROM events WHERE leg_id=? AND idempotency_key IS NOT NULL ORDER BY id DESC LIMIT 1",
                    (leg_id,),
                )
                row3 = cur3.fetchone()
                if row3 and row3[0]:
                    ikey = row3[0]
            open_legs.append(
                {
                    "trade_id": trade_id,
                    "leg_id": leg_id,
                    "state": etype,
                    "side": payload.get("side"),
                    "symbol": payload.get("symbol"),
                    "qty": payload.get("qty"),
                    "limit_price": payload.get("limit_price"),
                    "filled_qty": payload.get("filled_qty", 0),
                    "avg_price": payload.get("avg_price", 0.0),
                    "broker_order_id": boid,
                    "idempotency_key": ikey,
                }
            )
        return open_legs


# ---------------- compatibility helpers ----------------
def read_trades_between(
    start_dt: datetime, end_dt: datetime, path: str = "data/journal.sqlite"
) -> List[Dict[str, Any]]:
    """Return trades closed between ``start_dt`` and ``end_dt`` from the journal."""
    j = Journal.open(path)
    cur = j._exec(
        "SELECT ts_exit,pnl_R FROM trades WHERE ts_exit>=? AND ts_exit<=?",
        (start_dt.isoformat(), end_dt.isoformat()),
    )
    rows: List[Dict[str, Any]] = []
    for ts_exit, pnl_R in cur.fetchall():
        rows.append(
            {
                "ts_close": datetime.fromisoformat(ts_exit),
                "pnl_R": float(pnl_R),
            }
        )
    return rows
