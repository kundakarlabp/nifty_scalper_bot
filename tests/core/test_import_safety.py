from __future__ import annotations

import importlib
import sys


def test_core_package_import_does_not_eagerly_import_app_module() -> None:
    sys.modules.pop("nifty_scalper_bot.core", None)
    sys.modules.pop("nifty_scalper_bot.core.app", None)

    importlib.import_module("nifty_scalper_bot.core")

    assert "nifty_scalper_bot.core.app" not in sys.modules


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
