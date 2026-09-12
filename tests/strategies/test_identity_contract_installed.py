"""Identity/runtime contracts are native owners, not import-time patches."""

from __future__ import annotations

import inspect

import nifty_scalper_bot.strategies as strategies
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_signal_identity_is_native_property() -> None:
    source = inspect.getsource(Signal.deterministic_id.fget)
    assert "deterministic_signal_id" in source
    assert isinstance(Signal.deterministic_id, property)
    assert not hasattr(Signal, "_stable_setup_identity_patch")


def test_elite_signal_finalization_is_native() -> None:
    source = inspect.getsource(EliteStrategy.generate_signal)
    assert "finalize_signal_observability" in source
    assert "_elite_signal_observability_patch" not in source


def test_runtime_context_normalization_is_native() -> None:
    source = inspect.getsource(IndicatorEngine.set_runtime_context)
    assert "normalise_live_direction_context" in source
    assert "_live_direction_contract_installed" not in source


def test_strategy_package_has_no_runtime_patch_installer() -> None:
    source = inspect.getsource(strategies)
    assert "apply_patches" not in source
    assert "install_indicator_runtime_context_contract" not in source


def test_elite_exports_are_available() -> None:
    assert strategies.__all__
