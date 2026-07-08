"""Operator Telegram command aliases loaded at notifications import time."""

from __future__ import annotations

import importlib

_PATCH_APPLIED = False


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _operator = importlib.import_module("nifty_scalper_bot.notifications.operator_telegram")

    existing = {spec.name for spec in _operator.OPERATOR_COMMANDS}
    if "flat" not in existing:
        flat_spec = _operator.CommandSpec(
            "flat",
            "alias for confirmed flatten of bot-owned open positions",
            _operator.cmd_flatten,
            "Control",
            "confirmed-destructive",
        )
        names = [spec.name for spec in _operator.OPERATOR_COMMANDS]
        try:
            insert_at = names.index("flatten") + 1
        except ValueError:
            insert_at = len(_operator.OPERATOR_COMMANDS)
        _operator.OPERATOR_COMMANDS.insert(insert_at, flat_spec)
        _operator.OPERATOR_COMMAND_NAMES = tuple(spec.name for spec in _operator.OPERATOR_COMMANDS)
    _PATCH_APPLIED = True


__all__ = ["apply_patches"]
