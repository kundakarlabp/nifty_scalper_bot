"""Operator Telegram command aliases loaded at notifications import time."""

from __future__ import annotations

_PATCH_APPLIED = False


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.notifications import operator_telegram as _operator

    existing = {spec.name for spec in _operator.OPERATOR_COMMANDS}
    if "flat" not in existing:
        _operator.OPERATOR_COMMANDS.append(
            _operator.CommandSpec(
                "flat",
                "alias for confirmed flatten of bot-owned open positions",
                _operator.cmd_flatten,
                "Control",
                "confirmed-destructive",
            )
        )
        _operator.OPERATOR_COMMAND_NAMES = tuple(spec.name for spec in _operator.OPERATOR_COMMANDS)
    _PATCH_APPLIED = True


__all__ = ["apply_patches"]
