"""Regression tests for :mod:`nifty_scalper_bot.core.app` imports."""

from __future__ import annotations

import asyncio
from importlib import import_module


def test_core_app_exposes_asyncio_module() -> None:
    """Ensure ``asyncio`` remains imported for runtime entrypoints."""

    module = import_module("nifty_scalper_bot.core.app")
    assert getattr(module, "asyncio", None) is asyncio
