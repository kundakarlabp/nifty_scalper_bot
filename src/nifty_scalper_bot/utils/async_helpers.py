import asyncio
from typing import Any, Awaitable

def run_sync(coro: Awaitable[Any]) -> Any:
    """Safely runs a coroutine synchronously, blocking the current thread."""
    try:
        # Check if an event loop is already running (e.g., if called from another async function's thread)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running, run the coroutine in its own new loop (the simple case)
        return asyncio.run(coro)
    else:
        # If a loop is running, we must run the coroutine on the existing loop
        # using run_until_complete() and block the current synchronous code.
        return loop.run_until_complete(coro)
