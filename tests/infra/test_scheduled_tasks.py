from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import pytest

from nifty_scalper_bot.infra.scheduled_tasks import (
    run_archive_rotation,
    run_periodic_task,
    start_background_tasks,
    start_trade_replication_task,
)


class _DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, object]]] = []

    def info(self, message: str, *, extra: dict[str, object]) -> None:
        self.messages.append((message, extra))


class _DummyOrderManager:
    def __init__(self, history_path: Any | None) -> None:
        self._history_persist_path = history_path

    def persist_history_batch(self) -> None:  # pragma: no cover - simple stub
        return None


@pytest.mark.asyncio
async def test_run_periodic_task_executes_async_callable() -> None:
    call_count = 0

    async def task_fn() -> None:
        nonlocal call_count
        call_count += 1

    task = asyncio.create_task(run_periodic_task(task_fn, 0.0, "unit"))
    await asyncio.sleep(0.01)
    assert call_count >= 1
    task.cancel()
    await task


@pytest.mark.asyncio
async def test_run_periodic_task_executes_sync_callable() -> None:
    call_count = 0

    def task_fn() -> None:
        nonlocal call_count
        call_count += 1

    task = asyncio.create_task(run_periodic_task(task_fn, 0.0, "unit-sync"))
    await asyncio.sleep(0.01)
    assert call_count >= 1
    task.cancel()
    await task


@pytest.mark.asyncio
async def test_run_archive_rotation_handles_missing_path(tmp_path: Any) -> None:
    order_manager = _DummyOrderManager(history_path=None)
    await run_archive_rotation(order_manager, max_age_days=1)


@pytest.mark.asyncio
async def test_run_archive_rotation_invokes_rotate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    recorded: dict[str, Any] = {}

    def fake_rotate(path: Any, max_age_days: int) -> None:
        recorded["path"] = path
        recorded["max_age_days"] = max_age_days

    monkeypatch.setattr(
        "nifty_scalper_bot.infra.scheduled_tasks.rotate_order_history_archive",
        fake_rotate,
    )

    history_path = tmp_path / "orders.jsonl"
    manager = _DummyOrderManager(history_path=history_path)
    await run_archive_rotation(manager, max_age_days=12)

    assert recorded == {"path": history_path, "max_age_days": 12}


def test_start_background_tasks_creates_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class _FakeTask:
        def __init__(self, coro: Awaitable[Any]) -> None:
            self.coro = coro

    def fake_create_task(coro: Awaitable[Any]) -> _FakeTask:
        task = _FakeTask(coro)
        created.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    history_path = "memory"
    manager = _DummyOrderManager(history_path=history_path)
    logger = _DummyLogger()

    tasks = start_background_tasks(manager, logger)

    assert len(tasks) == 2
    assert created == tasks
    assert logger.messages[-1][0] == "Scheduled maintenance tasks created"
    assert logger.messages[-1][1]["count"] == 2


def test_start_trade_replication_task_is_independent_of_execution_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    created: list[Any] = []

    class _FakeTask:
        def __init__(self, coro: Awaitable[Any]) -> None:
            self.coro = coro

    class _Replicator:
        def replicate_once(self) -> dict[str, int]:
            return {"events": 0, "ledger": 0, "checkpoint": 0}

    def fake_create_task(coro: Awaitable[Any]) -> _FakeTask:
        task = _FakeTask(coro)
        created.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(
        "nifty_scalper_bot.infra.scheduled_tasks.build_supabase_trade_replicator",
        lambda _path: _Replicator(),
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.infra.scheduled_tasks.replication_interval_seconds",
        lambda: 30.0,
    )

    task = start_trade_replication_task(tmp_path / "trades.db")

    assert task is created[0]
    assert len(created) == 1


@pytest.mark.asyncio
async def test_run_periodic_task_logs_task_name_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, tuple[Any, ...]]] = []

    class _Logger:
        def debug(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def info(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def error(self, message: str, *args: Any, **_kwargs: Any) -> None:
            messages.append((message, args))

    monkeypatch.setattr("nifty_scalper_bot.infra.scheduled_tasks.LOGGER", _Logger())

    def fail() -> None:
        raise RuntimeError("remote unavailable")

    task = asyncio.create_task(run_periodic_task(fail, 0.0, "replicate_trade"))
    await asyncio.sleep(0.01)
    task.cancel()
    await task

    assert messages[-1][0] == "Periodic task failed name=%s error=%s"
    assert messages[-1][1][0] == "replicate_trade"
    assert str(messages[-1][1][1]) == "remote unavailable"
