"""Notification facade — decouples bot from Telegram."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.order_state import Position
    from ..strategies.signal import Signal

log = logging.getLogger(__name__)


class Notifier:
    """Abstract notification facade. Delegates to Telegram if configured."""

    def __init__(self, telegram_bot=None) -> None:
        self._tg = telegram_bot

    async def send_fill_alert(self, position: "Position", signal: "Signal") -> None:
        msg = (
            f"✅ *FILL* | {signal.direction.value}\n"
            f"Symbol: `{position.symbol}`\n"
            f"Entry: ₹{position.entry_price:.2f} × {signal.quantity} lots\n"
            f"SL: ₹{signal.sl_price:.2f} | TP1: ₹{signal.tp1_price:.2f}\n"
            f"Strategy: {signal.strategy_name} | Conf: {signal.confidence:.0%}"
        )
        await self._send(msg)

    async def send_exit_alert(self, position: "Position", exit_reason: str, pnl: float) -> None:
        emoji = "💚" if pnl > 0 else "🔴"
        msg = (
            f"{emoji} *EXIT* [{exit_reason}] | {position.symbol}\n"
            f"P&L: ₹{pnl:+.2f}"
        )
        await self._send(msg)

    async def send_daily_summary(self, stats: dict) -> None:
        msg = (
            f"📊 *Daily Summary*\n"
            f"Trades: {stats.get('trades', 0)}\n"
            f"Net P&L: ₹{stats.get('net_pnl', 0):+.2f}\n"
            f"Win Rate: {stats.get('win_rate', 0):.0%}\n"
            f"Max Drawdown: ₹{stats.get('max_drawdown', 0):.2f}"
        )
        await self._send(msg)

    async def send_sl_hit_alert(self, position: "Position") -> None:
        msg = f"🛑 *SL HIT* | {position.symbol} | Loss: ₹{position.pnl_unrealized:.2f}"
        await self._send(msg)

    async def send_error(self, message: str) -> None:
        await self._send(f"⚠️ *ERROR*\n{message}")

    async def send_text(self, message: str) -> None:
        await self._send(message)

    async def _send(self, message: str) -> None:
        if self._tg:
            try:
                await self._tg.send_message(message)
            except Exception as e:
                log.error("Telegram send failed: %s", e)
        else:
            log.info("NOTIFY: %s", message[:100])
