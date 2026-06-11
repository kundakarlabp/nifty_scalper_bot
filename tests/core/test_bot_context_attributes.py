"""Permanent guard against BotContext AttributeError crashes.

BotContext is @dataclass(slots=True): assigning an undeclared attribute at
runtime raises AttributeError (e.g. the LIVE_READINESS_REARM_LOOP_CRASHED
incident on 2026-06-12). This test statically audits every `ctx.<attr> =`
assignment in core/app.py and fails if any attribute is neither declared on
the BotContext dataclass nor registered in BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS.
Any future edit that reintroduces an ad-hoc ctx attribute fails CI here
instead of crashing a production loop.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nifty_scalper_bot.core.app as app_module
from nifty_scalper_bot.core.app import BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS, BotContext


def _app_source() -> str:
    return Path(app_module.__file__).read_text()


def test_every_ctx_attribute_write_is_declared() -> None:
    src = _app_source()
    tree = ast.parse(src)
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "BotContext":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    declared.add(stmt.target.id)
    defaults = set(BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS.keys())
    writes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ctx"
                ):
                    writes.add(target.attr)
    undeclared = sorted(writes - declared - defaults)
    assert not undeclared, (
        f"Undeclared ctx attribute writes (would raise AttributeError on the "
        f"slots=True BotContext): {undeclared}. Declare the field on BotContext "
        f"and register it in BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS, or use local state."
    )


def test_basket_build_pending_spot_ltp_is_declared_field() -> None:
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(BotContext)}
    assert "basket_build_pending_spot_ltp" in field_names


def test_rearm_loop_does_not_write_private_ctx_flags() -> None:
    # The crashed code path used ctx._rearm_sleep_logged; the flag must be
    # loop-local now and never assigned on ctx.
    src = _app_source()
    assert not re.search(r"ctx\._rearm_sleep_logged\s*=", src)
