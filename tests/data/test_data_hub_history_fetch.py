from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub


class _HistoryMdmStub:
    def __init__(
        self, rows: list[dict[str, Any]], exc: Exception | None = None
    ) -> None:
        """Create history stub. Args: rows, exc. Returns: None. Raises: None."""
        self.rows = rows
        self.exc = exc

    async def fetch_history(
        self, symbol: str, interval: str, days: int = 3
    ) -> list[dict[str, Any]]:
        """Return stub rows. Args: inputs. Returns: rows. Raises: Exception."""
        if self.exc is not None:
            raise self.exc
        return list(self.rows)


class _ResolverStub:
    def resolve_token_to_symbol(self, token: int) -> str | None:
        """Resolve token. Args: token. Returns: None. Raises: None."""
        return None


def test_fetch_history_drops_incomplete_current_minute_bar() -> None:
    now = datetime.now(timezone.utc).replace(second=30, microsecond=0)
    closed_ts = now - timedelta(minutes=2)
    open_ts = now.replace(second=0, microsecond=0)
    rows = [
        {
            "timestamp": closed_ts,
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 10,
        },
        {
            "timestamp": open_ts,
            "open": 101,
            "high": 103,
            "low": 100,
            "close": 102,
            "volume": 5,
        },
    ]
    hub = DataHub(_HistoryMdmStub(rows), _ResolverStub(), options_only=True)

    result = asyncio.run(hub.fetch_history("NIFTY", "minute", 1))

    assert len(result) == 1
    assert result[0]["timestamp"] == closed_ts.replace(second=0, microsecond=0)


def test_fetch_history_uses_stale_cache_when_fetch_fails() -> None:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    seed_rows = [
        {
            "timestamp": now - timedelta(minutes=3),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100.5,
            "volume": 20,
        }
    ]
    mdm = _HistoryMdmStub(seed_rows)
    hub = DataHub(mdm, _ResolverStub(), options_only=True)

    seeded = asyncio.run(hub.fetch_history("NIFTY", "minute", 1))
    assert len(seeded) == 1

    mdm.exc = RuntimeError("network down")
    fallback = asyncio.run(hub.fetch_history("NIFTY", "minute", 1))

    assert fallback == seeded
