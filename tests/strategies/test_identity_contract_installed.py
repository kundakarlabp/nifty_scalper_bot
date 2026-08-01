"""Identity/runtime contracts must be installed, not silently skipped (P2)."""

from __future__ import annotations

import inspect

import nifty_scalper_bot.strategies as strategies
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_signal_identity_patch_is_installed_at_import() -> None:
    assert getattr(Signal, "_stable_setup_identity_patch", False) is True
    assert isinstance(Signal.deterministic_id, property)


def test_contract_installation_is_not_swallowed() -> None:
    source = inspect.getsource(strategies)
    install_index = source.index("_apply_signal_identity_patches()")
    preceding = source[:install_index]
    # The historical anti-pattern: a bare try/except: pass around the install.
    assert "except Exception:\n    pass" not in preceding


def test_elite_exports_are_available() -> None:
    assert strategies.__all__
