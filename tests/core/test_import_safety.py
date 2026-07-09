from __future__ import annotations

import importlib
import sys


def _reset_core_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.core" or name.startswith("nifty_scalper_bot.core.app"):
            sys.modules.pop(name, None)


def test_core_package_import_does_not_eagerly_import_app_module() -> None:
    _reset_core_modules()

    importlib.import_module("nifty_scalper_bot.core")

    assert "nifty_scalper_bot.core.app" not in sys.modules


def test_direct_core_app_import_applies_polling_patch_without_lazy_getattr() -> None:
    _reset_core_modules()

    app_module = importlib.import_module("nifty_scalper_bot.core.app")

    assert getattr(app_module, "_polling_failover_runtime_patch_installed", False) is True
    assert callable(getattr(app_module, "_polling_failover_supervisor_iteration", None))


def test_core_lazy_app_resolution_applies_polling_patch() -> None:
    core = importlib.import_module("nifty_scalper_bot.core")
    app_module = core.__getattr__("app")

    assert getattr(app_module, "_polling_failover_runtime_patch_installed", False) is True
    assert callable(getattr(app_module, "_polling_failover_supervisor_iteration", None))


def test_nifty_scalper_app_lazy_resolution_applies_app_patches() -> None:
    core = importlib.import_module("nifty_scalper_bot.core")
    app_cls = core.NiftyScalperApp
    app_module = sys.modules.get("nifty_scalper_bot.core.app")

    assert app_cls is getattr(app_module, "NiftyScalperApp")
    assert getattr(app_module, "_polling_failover_runtime_patch_installed", False) is True
