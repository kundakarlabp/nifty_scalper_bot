from __future__ import annotations

import importlib
import sys

CORE_HOOK_ATTR = "_nifty_scalper_core_app_patch_hook"
DATA_HOOK_ATTR = "_nifty_scalper_datahub_synthetic_guard_hook"


def _hook_count(attr: str) -> int:
    return sum(1 for finder in sys.meta_path if getattr(finder, attr, False))


def _reset_core_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.core" or name.startswith(
            "nifty_scalper_bot.core.app"
        ):
            sys.modules.pop(name, None)


def _reset_data_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.data" or name.startswith(
            "nifty_scalper_bot.data.data_hub"
        ):
            sys.modules.pop(name, None)


def test_core_package_reloads_never_install_core_app_import_hook() -> None:
    _reset_core_modules()
    assert _hook_count(CORE_HOOK_ATTR) == 0

    core = importlib.import_module("nifty_scalper_bot.core")
    assert _hook_count(CORE_HOOK_ATTR) == 0

    for _ in range(3):
        core = importlib.reload(core)
        assert _hook_count(CORE_HOOK_ATTR) == 0

    for _ in range(3):
        _reset_core_modules()
        importlib.import_module("nifty_scalper_bot.core")
        assert _hook_count(CORE_HOOK_ATTR) == 0


def test_core_app_direct_import_bootstraps_hardening_without_import_hook() -> None:
    _reset_core_modules()
    importlib.import_module("nifty_scalper_bot.core")

    app_module = importlib.import_module("nifty_scalper_bot.core.app")

    assert _hook_count(CORE_HOOK_ATTR) == 0
    assert getattr(app_module, "_native_runtime_hardening_bootstrap", False) is True
    supervisor = getattr(app_module, "_polling_failover_supervisor_iteration", None)
    assert callable(supervisor)
    assert getattr(supervisor, "__module__", None) == "nifty_scalper_bot.core.app"
    assert not hasattr(app_module, "_polling_failover_runtime_patch_installed")


def test_data_package_reloads_never_install_datahub_import_hook() -> None:
    _reset_data_modules()
    before = _hook_count(DATA_HOOK_ATTR)

    data_pkg = importlib.import_module("nifty_scalper_bot.data")

    assert _hook_count(DATA_HOOK_ATTR) == before == 0
    for _ in range(3):
        data_pkg = importlib.reload(data_pkg)
        assert _hook_count(DATA_HOOK_ATTR) == 0


def test_direct_datahub_import_uses_native_guard_without_hook() -> None:
    _reset_data_modules()
    importlib.import_module("nifty_scalper_bot.data")

    datahub_module = importlib.import_module("nifty_scalper_bot.data.data_hub")

    assert _hook_count(DATA_HOOK_ATTR) == 0
    assert (
        getattr(
            datahub_module.DataHub,
            "_synthetic_timestamp_guard_installed",
            False,
        )
        is True
    )
    assert datahub_module.DataHub.store_quote.__module__ == (
        "nifty_scalper_bot.data.data_hub"
    )
