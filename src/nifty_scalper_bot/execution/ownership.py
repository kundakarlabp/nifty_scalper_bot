"""Explicit execution ownership bindings."""

from __future__ import annotations

from contextlib import suppress

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


class BoundBracketManager(RuntimeBracketManager):
    """Bracket authority that configures the OrderManager native entry gate."""

    def _install_unresolved_exit_entry_guard(self) -> None:
        order_manager = getattr(self, "order_manager", None)
        setter = getattr(order_manager, "set_unresolved_exit_provider", None)
        if callable(setter):
            setter(self)
            with suppress(Exception):
                from nifty_scalper_bot.execution import legacy_bracket_manager as legacy

                legacy.LOGGER.info(
                    "UNRESOLVED_EXIT_NATIVE_GATE_BOUND",
                    extra={"event": "UNRESOLVED_EXIT_NATIVE_GATE_BOUND"},
                )
            return
        super()._install_unresolved_exit_entry_guard()


__all__ = ["BoundBracketManager"]
