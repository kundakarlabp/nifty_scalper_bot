"""Trade recording."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .db import AsyncDB


@dataclass
class Trade:
    trade_id: str
    strategy_name: str
    symbol: str
    underlying: str
    option_type: str
    strike: float
    expiry: date
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    gross_pnl: Optional[float]
    brokerage: Optional[float]
    net_pnl: Optional[float]
    sl_price: float
    tp1_price: float
    tp2_price: float
    exit_reason: Optional[str]
    regime_at_entry: str
    confidence_at_entry: float
    entry_time: datetime
    exit_time: Optional[datetime]
    duration_minutes: Optional[float]
    r_multiple_achieved: Optional[float]
    is_winner: bool = False


class TradeLog:
    def __init__(self, db: AsyncDB) -> None:
        self._db = db

    async def insert_trade(self, trade: Trade) -> None:
        await self._db.execute(
            """INSERT INTO trades (trade_id, strategy_name, symbol, underlying, option_type,
               strike, expiry, direction, entry_price, exit_price, quantity, gross_pnl,
               brokerage, net_pnl, sl_price, tp1_price, tp2_price, exit_reason,
               regime_at_entry, confidence_at_entry, entry_time, exit_time,
               duration_minutes, r_multiple_achieved, is_winner)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade.trade_id, trade.strategy_name, trade.symbol, trade.underlying,
                trade.option_type, trade.strike, str(trade.expiry), trade.direction,
                trade.entry_price, trade.exit_price, trade.quantity, trade.gross_pnl,
                trade.brokerage, trade.net_pnl, trade.sl_price, trade.tp1_price,
                trade.tp2_price, trade.exit_reason, trade.regime_at_entry,
                trade.confidence_at_entry, trade.entry_time.isoformat(),
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.duration_minutes, trade.r_multiple_achieved, int(trade.is_winner),
            ),
        )

    async def update_exit(self, trade_id: str, exit_price: float, exit_reason: str, exit_time: datetime, net_pnl: float) -> None:
        await self._db.execute(
            """UPDATE trades SET exit_price=?, exit_reason=?, exit_time=?,
               net_pnl=?, is_winner=? WHERE trade_id=?""",
            (exit_price, exit_reason, exit_time.isoformat(), net_pnl, int(net_pnl > 0), trade_id),
        )

    async def get_open_trades(self) -> list[dict]:
        return await self._db.fetchall("SELECT * FROM trades WHERE exit_time IS NULL")
