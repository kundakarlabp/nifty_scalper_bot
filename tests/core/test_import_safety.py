from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
import types


def _reset_core_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.core" or name.startswith(
            "nifty_scalper_bot.core.app"
        ):
            sys.modules.pop(name, None)


def _assert_native_polling_owner(app_module) -> None:
    supervisor = getattr(app_module, "_polling_failover_supervisor_iteration", None)
    assert callable(supervisor)
    assert getattr(supervisor, "__module__", None) == "nifty_scalper_bot.core.app"
    assert not hasattr(app_module, "_polling_failover_runtime_patch_installed")
    assert (
        "quote_stale_ms"
        in inspect.signature(app_module._polling_fallback_degraded).parameters
    )


def test_core_package_import_does_not_eagerly_import_app_module() -> None:
    _reset_core_modules()

    importlib.import_module("nifty_scalper_bot.core")

    assert "nifty_scalper_bot.core.app" not in sys.modules


def test_direct_core_app_import_uses_native_polling_owner() -> None:
    _reset_core_modules()

    app_module = importlib.import_module("nifty_scalper_bot.core.app")

    _assert_native_polling_owner(app_module)


def test_core_lazy_app_resolution_preserves_native_polling_owner() -> None:
    core = importlib.import_module("nifty_scalper_bot.core")
    app_module = core.__getattr__("app")

    _assert_native_polling_owner(app_module)


def test_nifty_scalper_app_lazy_resolution_preserves_native_polling_owner() -> None:
    core = importlib.import_module("nifty_scalper_bot.core")
    app_cls = core.NiftyScalperApp
    app_module = sys.modules.get("nifty_scalper_bot.core.app")

    assert app_cls is getattr(app_module, "NiftyScalperApp")
    _assert_native_polling_owner(app_module)


def test_native_polling_iteration_accepts_recover_cooldown_and_async_stop() -> None:
    app_module = importlib.import_module("nifty_scalper_bot.core.app")
    stops: list[bool] = []

    class _Fallback:
        def is_running(self) -> bool:
            return True

        async def set_websocket_mode(self, _enabled: bool) -> None:
            return None

        async def stop(self) -> None:
            stops.append(True)

    ctx = types.SimpleNamespace(
        is_market_open_now=lambda: False,
        websocket_manager=None,
        market_data_manager=None,
        event_loop_lagging=False,
    )
    result = asyncio.run(
        app_module._polling_failover_supervisor_iteration(
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
    assert stops == [True]
