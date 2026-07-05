"""File purpose:
    Bind the canonical bracket authority to the canonical order-entry gate.

Key responsibilities:
    - Register the active bracket manager as the unresolved-exit provider.
    - Preserve a compatibility fallback for noncanonical external test doubles.
    - Resolve live-mode consistently before durable bracket-state enforcement.

Operational constraints:
    - Production wiring must use the native provider contract, not method replacement.
    - Protective exits must remain executable while new entries are blocked.
    - LIVE executions must never persist bracket state to ephemeral storage.
"""

from __future__ import annotations

from contextlib import suppress
import os

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager

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


class BoundBracketManager(RuntimeBracketManager):
    """Bracket authority that configures the OrderManager native entry gate."""

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

    def _install_unresolved_exit_entry_guard(self) -> None:
        order_manager = getattr(self, "order_manager", None)
        setter = getattr(order_manager, "set_unresolved_exit_provider", None)
        if callable(setter):
            setter(self)
            with suppress(Exception):
                from nifty_scalper_bot.execution import bracket_core

                bracket_core.LOGGER.info(
                    "UNRESOLVED_EXIT_NATIVE_GATE_BOUND",
                    extra={"event": "UNRESOLVED_EXIT_NATIVE_GATE_BOUND"},
                )
            return
        super()._install_unresolved_exit_entry_guard()


__all__ = ["BoundBracketManager"]
