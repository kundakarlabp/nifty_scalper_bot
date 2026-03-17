import asyncio
from typing import Any, Awaitable


def run_sync(coro: Awaitable[Any]) -> Any:
    """Run coroutine from sync context safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.create_task(coro)
