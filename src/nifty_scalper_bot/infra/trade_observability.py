"""Non-blocking trade lifecycle telemetry primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """Canonical structured event shared by signal and execution telemetry."""

    event_name: str
    event_at: datetime
    trade_id: str | None = None
    signal_id: str | None = None
    trace_id: str | None = None
    symbol: str | None = None
    strategy: str | None = None
    reason_code: str | None = None
    build_sha: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """Return the persistence payload with canonical IST trading date."""
        row = asdict(self)
        row["event_at"] = self.event_at.isoformat()
        row["trading_date"] = self.event_at.astimezone(IST).date().isoformat()
        return row


__all__ = ["TradeEvent"]
