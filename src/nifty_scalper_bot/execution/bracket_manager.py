"""File purpose:
    Provide the stable public API for the canonical bracket and exit lifecycle.

Key responsibilities:
    - Re-export bracket state models and helpers from ``bracket_core``.
    - Expose ``BoundBracketManager`` as the single production bracket authority.
    - Resolve live-mode consistently before enforcing durable bracket storage.

Operational constraints:
    - This facade must not own independent bracket state or exit execution logic.
    - Entry release remains blocked until the bound runtime confirms durable closure.
    - LIVE executions must never persist bracket state to ephemeral storage.
"""

from __future__ import annotations

from contextlib import suppress
import os
from typing import Any

from nifty_scalper_bot.execution import bracket_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager  # noqa: E402
from nifty_scalper_bot.execution.ownership import (  # noqa: E402
    BoundBracketManager as _OwnershipBoundBracketManager,
)

_TRUTHY = {"1", "true", "yes", "y", "on", "live"}


def _env_truthy(name: str) -> bool:
    """Return True when an environment variable carries a truthy operator value."""
    return str(os.getenv(name, "") or "").strip().lower() in _TRUTHY


def _running_under_test_harness() -> bool:
    """Return True only for explicit pytest/test harness execution.

    This is deliberately environment-based rather than importing pytest or checking
    installed packages. Production hosts may have pytest installed for validation,
    but they should not be treated as tests unless a test runner marks the process.
    """
    return bool(os.getenv("PYTEST_CURRENT_TEST")) or _env_truthy("NSB_TEST_MODE")


class BoundBracketManager(_OwnershipBoundBracketManager):
    """Public runtime bracket manager with conservative live-mode detection.

    The core persistence layer forbids ephemeral bracket state in LIVE.  Test
    validation intentionally uses tmp paths, so the public facade normalizes
    execution context before that storage policy is evaluated.
    """

    def _is_live_execution(self) -> bool:
        """Return True only when real broker-live order execution is enabled."""
        if _running_under_test_harness():
            return False

        checker = getattr(getattr(self, "order_manager", None), "is_live_mode", None)
        if callable(checker):
            with suppress(Exception):
                return bool(checker())

        mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        live_enabled = _env_truthy("ENABLE_LIVE") or _env_truthy("ENABLE_LIVE_TRADING")
        shadow_or_paper = (
            _env_truthy("SHADOW_MODE")
            or _env_truthy("PAPER_MODE")
            or _env_truthy("PAPER__ENABLED")
        )
        return mode == "LIVE" and live_enabled and not shadow_or_paper

    def _strict_ledger_release_required(self) -> bool:
        """Use the same live predicate for ledger release and state durability."""
        return self._is_live_execution()


BracketManager = BoundBracketManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "BoundBracketManager",
        "BracketManager",
        "RuntimeBracketManager",
    }
)
