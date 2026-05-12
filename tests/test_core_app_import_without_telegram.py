"""Import safety test when telegram dependency is unavailable."""

from __future__ import annotations

import builtins
from importlib import import_module


def test_core_app_import_without_telegram_monkeypatched(monkeypatch) -> None:
    """Ensure core app import succeeds without telegram package. Args: monkeypatch. Returns: none. Raises: none."""

    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith('telegram'):
            raise ImportError('telegram unavailable in test')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', _guarded_import)
    module = import_module('nifty_scalper_bot.core.app')
    assert module is not None
