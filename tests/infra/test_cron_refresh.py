"""Tests for the instrument refresh scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.data import InstrumentUniverseStatus
from nifty_scalper_bot.infra import cron_refresh


def _write_sample_csv(csv_path: Path) -> None:
    csv_path.write_text(
        (
            "instrument_token,exchange,tradingsymbol,lot_size,expiry,strike,instrument_type\n"
            "101,NFO,NIFTY25OCT26000CE,50,2025-10-30,26000.0,CE\n"
            "102,NFO,NIFTY25OCT26000PE,50,2025-10-30,26000.0,PE\n"
        )
    )


def test_schedule_instrument_refresh_skips_without_config() -> None:
    """Scheduler should skip when configuration is missing or incomplete."""

    resolver = SimpleNamespace()
    settings = SimpleNamespace()

    assert cron_refresh.schedule_instrument_refresh(settings, resolver) is None

    settings.instruments = SimpleNamespace(csv_path=None, refresh_cron_enabled=True)
    assert cron_refresh.schedule_instrument_refresh(settings, resolver) is None

    settings.instruments.csv_path = Path("/nonexistent.csv")
    settings.instruments.refresh_cron_enabled = False
    assert cron_refresh.schedule_instrument_refresh(settings, resolver) is None


@pytest.mark.asyncio
async def test_schedule_instrument_refresh_runs_and_updates_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduler should refresh cache and update resolver status once per cycle."""

    csv_path = tmp_path / "instruments.csv"
    _write_sample_csv(csv_path)
    db_path = tmp_path / "cache.sqlite"

    instruments = SimpleNamespace(
        refresh_cron_enabled=True,
        csv_path=csv_path,
        db_path=db_path,
    )
    settings = SimpleNamespace(instruments=instruments)

    class DummyResolver:
        def __init__(self) -> None:
            self.calls = 0
            self.rows: list[dict[str, int | str]] = []

        def warm_from_broker_dump(self, rows: list[dict[str, int | str]]) -> None:
            self.calls += 1
            self.rows = list(rows)

    resolver = DummyResolver()
    state = InstrumentUniverseStatus()

    monkeypatch.setattr(
        cron_refresh,
        "_seconds_until_run",
        lambda *_args, **_kwargs: 0.0,
    )

    sleep_calls = 0

    async def fake_sleep(_delay: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(cron_refresh.asyncio, "sleep", fake_sleep)

    original_create_task = asyncio.create_task
    captured_task: asyncio.Task[None] | None = None

    def fake_create_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        nonlocal captured_task
        captured_task = original_create_task(coro)
        return captured_task

    monkeypatch.setattr(cron_refresh.asyncio, "create_task", fake_create_task)

    task = cron_refresh.schedule_instrument_refresh(settings, resolver, state=state)
    assert task is captured_task
    assert task is not None

    with pytest.raises(asyncio.CancelledError):
        await task

    assert resolver.calls == 1
    assert resolver.rows
    assert state.tokens == len(resolver.rows)
    assert state.options == len(resolver.rows)
    assert state.last_source == "cron"
    assert state.last_path == str(db_path)
    assert state.last_refresh is not None
    assert state.last_refresh.tzinfo is timezone.utc
