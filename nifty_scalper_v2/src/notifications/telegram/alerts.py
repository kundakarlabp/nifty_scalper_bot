"""Telegram alert formatters."""

from __future__ import annotations


def format_fill(symbol: str, direction: str, entry: float, sl: float, tp1: float, qty: int, strategy: str, confidence: float) -> str:
    return (
        f"✅ *FILL* | {direction}\n"
        f"`{symbol}` × {qty} lots\n"
        f"Entry: ₹{entry:.2f} | SL: ₹{sl:.2f} | TP1: ₹{tp1:.2f}\n"
        f"{strategy} | {confidence:.0%}"
    )


def format_exit(symbol: str, reason: str, pnl: float) -> str:
    emoji = "💚" if pnl > 0 else "🔴"
    return f"{emoji} *EXIT* [{reason}] `{symbol}` P&L: ₹{pnl:+.2f}"


def format_daily_summary(date_str: str, trades: int, net_pnl: float, win_rate: float, max_dd: float) -> str:
    return (
        f"📊 *Daily Summary — {date_str}*\n"
        f"Trades: {trades} | Win Rate: {win_rate:.0%}\n"
        f"Net P&L: ₹{net_pnl:+,.2f}\n"
        f"Max Drawdown: ₹{max_dd:,.2f}"
    )


def format_status(mode: str, positions: int, daily_pnl: float, regime: str, vix: float, paused: bool) -> str:
    status = "⏸ PAUSED" if paused else "▶ RUNNING"
    return (
        f"🤖 *NiftyScalperBot v2* — {status}\n"
        f"Mode: {mode} | Regime: {regime} | VIX: {vix:.1f}\n"
        f"Positions: {positions} | Daily P&L: ₹{daily_pnl:+,.2f}"
    )
