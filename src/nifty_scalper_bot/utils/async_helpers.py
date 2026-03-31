import asyncio
from typing import Any, Awaitable

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def safe_task(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    """Create guarded asyncio task. Args: coro. Returns: task. Raises: RuntimeError."""

    async def wrapper() -> Any:
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 - task guard
            LOGGER.exception("Unhandled async error: %s", exc)
            return None

    return asyncio.create_task(wrapper())


def run_sync(coro: Awaitable[Any]) -> Any:
    """Run coroutine from sync context safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return safe_task(coro)
