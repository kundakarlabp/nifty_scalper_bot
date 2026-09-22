from __future__ import annotations

import importlib
import sys

CORE_HOOK_ATTR = "_nifty_scalper_core_app_patch_hook"
DATA_HOOK_ATTR = "_nifty_scalper_datahub_synthetic_guard_hook"


def _hook_count(attr: str) -> int:
    return sum(1 for finder in sys.meta_path if getattr(finder, attr, False))


def _reset_core_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.core" or name.startswith("nifty_scalper_bot.core.app"):
            sys.modules.pop(name, None)


def _reset_data_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.data" or name.startswith("nifty_scalper_bot.data.data_hub"):
            sys.modules.pop(name, None)


def test_core_app_import_hook_is_not_duplicated_by_repeated_imports() -> None:
    _reset_core_modules()
    before = _hook_count(CORE_HOOK_ATTR)

    core = importlib.import_module("nifty_scalper_bot.core")
    first = _hook_count(CORE_HOOK_ATTR)

    assert first == max(before, 1)
    assert first <= 1

    for _ in range(3):
        core = importlib.reload(core)
        assert _hook_count(CORE_HOOK_ATTR) == first

    for _ in range(3):
        _reset_core_modules()
        importlib.import_module("nifty_scalper_bot.core")
        assert _hook_count(CORE_HOOK_ATTR) == first


def test_core_app_direct_import_reuses_single_hook_and_still_patches_app() -> None:
    _reset_core_modules()
    importlib.import_module("nifty_scalper_bot.core")
    hook_count = _hook_count(CORE_HOOK_ATTR)

    app_module = importlib.import_module("nifty_scalper_bot.core.app")

    assert _hook_count(CORE_HOOK_ATTR) == hook_count == 1
    assert getattr(app_module, "_polling_failover_runtime_patch_installed", False) is True
    assert callable(getattr(app_module, "_polling_failover_supervisor_iteration", None))


def test_datahub_import_hook_is_not_duplicated_by_repeated_imports() -> None:
    _reset_data_modules()
    before = _hook_count(DATA_HOOK_ATTR)

    data_pkg = importlib.import_module("nifty_scalper_bot.data")
    first = _hook_count(DATA_HOOK_ATTR)

    assert first == max(before, 1)
    assert first <= 1

    for _ in range(3):
        data_pkg = importlib.reload(data_pkg)
        assert _hook_count(DATA_HOOK_ATTR) == first

    for _ in range(3):
        _reset_data_modules()
        importlib.import_module("nifty_scalper_bot.data")
        assert _hook_count(DATA_HOOK_ATTR) == first


def test_direct_datahub_import_reuses_single_hook_and_still_patches_datahub() -> None:
    _reset_data_modules()
    importlib.import_module("nifty_scalper_bot.data")
    hook_count = _hook_count(DATA_HOOK_ATTR)

    datahub_module = importlib.import_module("nifty_scalper_bot.data.data_hub")

    assert _hook_count(DATA_HOOK_ATTR) == hook_count == 1
    assert getattr(datahub_module.DataHub, "_synthetic_timestamp_guard_installed", False) is True
