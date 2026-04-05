"""Analytics queries."""

from __future__ import annotations

from datetime import date

from .db import AsyncDB


class JournalQueries:
    def __init__(self, db: AsyncDB) -> None:
        self._db = db

    async def get_daily_pnl(self, for_date: date | None = None) -> float:
        d = (for_date or date.today()).isoformat()
        row = await self._db.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE DATE(entry_time)=? AND exit_time IS NOT NULL",
            (d,),
        )
        return float(row["total"] or 0.0) if row else 0.0

    async def get_win_rate(self, days: int = 30) -> float:
        rows = await self._db.fetchall(
            "SELECT is_winner FROM trades WHERE exit_time IS NOT NULL AND DATE(entry_time) >= DATE('now', ?)",
            (f"-{days} days",),
        )
        if not rows:
            return 0.0
        wins = sum(1 for r in rows if r["is_winner"])
        return wins / len(rows)

    async def get_trade_count_today(self) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) as cnt FROM trades WHERE DATE(entry_time)=DATE('now')",
        )
        return int(row["cnt"]) if row else 0

    async def get_strategy_performance(self, strategy_name: str, days: int = 30) -> dict:
        rows = await self._db.fetchall(
            """SELECT is_winner, net_pnl FROM trades
               WHERE strategy_name=? AND exit_time IS NOT NULL
               AND DATE(entry_time) >= DATE('now', ?)""",
            (strategy_name, f"-{days} days"),
        )
        if not rows:
            return {"signals": 0, "win_rate": 0.0, "total_pnl": 0.0}
        wins = sum(1 for r in rows if r["is_winner"])
        return {
            "signals": len(rows),
            "win_rate": wins / len(rows),
            "total_pnl": sum(r["net_pnl"] or 0 for r in rows),
        }

    async def get_max_drawdown(self, days: int = 30) -> float:
        rows = await self._db.fetchall(
            """SELECT net_pnl FROM trades WHERE exit_time IS NOT NULL
               AND DATE(entry_time) >= DATE('now', ?) ORDER BY exit_time""",
            (f"-{days} days",),
        )
        if not rows:
            return 0.0
        peak = 0.0
        running = 0.0
        max_dd = 0.0
        for r in rows:
            running += float(r["net_pnl"] or 0)
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        return max_dd
