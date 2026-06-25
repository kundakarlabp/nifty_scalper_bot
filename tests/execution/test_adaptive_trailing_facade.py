from __future__ import annotations

import importlib

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution import adaptive_trailing
from nifty_scalper_bot.execution import adaptive_trailing_legacy
from nifty_scalper_bot.execution.hardened_adaptive_trailing import HardenedAdaptiveTrailingController


def test_adaptive_controller_identity_is_stable() -> None:
    controller = adaptive_trailing.AdaptiveTrailingController
    assert controller is HardenedAdaptiveTrailingController
    assert execution.AdaptiveTrailingController is controller
    assert adaptive_trailing.LegacyAdaptiveTrailingController is adaptive_trailing_legacy.AdaptiveTrailingController
    assert issubclass(controller, adaptive_trailing.LegacyAdaptiveTrailingController)
    assert adaptive_trailing.TrailingSpec is adaptive_trailing_legacy.TrailingSpec


def test_package_import_does_not_replace_adaptive_controller() -> None:
    before = adaptive_trailing.AdaptiveTrailingController
    package = importlib.import_module("nifty_scalper_bot.execution")
    assert package.AdaptiveTrailingController is before
    assert adaptive_trailing.AdaptiveTrailingController is before
