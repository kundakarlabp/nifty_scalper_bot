"""Health check data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class HealthStatus:
    status: str            # "healthy" | "degraded" | "starting"
    uptime_seconds: float
    mode: str
    open_positions: int
    daily_pnl: float
    last_tick_at: Optional[datetime]
    ws_connected: bool
    warmup_complete: bool
    version: str = "2.0.0"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "uptime_seconds": self.uptime_seconds,
            "mode": self.mode,
            "open_positions": self.open_positions,
            "daily_pnl": self.daily_pnl,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "ws_connected": self.ws_connected,
            "warmup_complete": self.warmup_complete,
            "version": self.version,
        }
