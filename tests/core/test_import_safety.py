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


def test_installed_polling_failover_iteration_accepts_recover_cooldown() -> None:
    """2026-07-09 incident: the runtime patch replaced core.app's supervisor
    iteration with a wrapper that rejected recover_cooldown, so every
    supervisor loop iteration raised TypeError (213x in one session) and the
    WS->REST polling failover safety net was dead. The installed wrapper must
    accept the exact call shape used by core.app's supervisor loop and honor
    the anti-flap recover cooldown."""
    import asyncio
    import types

    from nifty_scalper_bot.core import polling_failover_runtime as pfr

    app_module = types.SimpleNamespace()
    pfr.apply_app_patch(app_module)
    installed = app_module._polling_failover_supervisor_iteration

    stops: list = []

    class _Fallback:
        def is_running(self) -> bool:
            return True

        async def stop(self) -> None:
            stops.append(True)

    ctx = types.SimpleNamespace(
        websocket_manager=None, market_data_manager=None, event_loop_lagging=False
    )
    result = asyncio.run(
        installed(
            ctx,
            _Fallback(),
            quote_stale_ms=5000.0,
            degraded_since=None,
            recovered_since=None,
            activate_after=5.0,
            recover_cooldown=10.0,
        )
    )
    assert isinstance(result, tuple) and len(result) == 2
