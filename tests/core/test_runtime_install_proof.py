from __future__ import annotations

import sys
from types import SimpleNamespace

from nifty_scalper_bot.core.runtime_install_proof import build_runtime_install_proof


class _LegacyCoreHook:
    _nifty_scalper_core_app_patch_hook = True


class _PlainFinder:
    pass


async def _native_polling_supervisor(*_args, **_kwargs):
    return None, None


_native_polling_supervisor.__module__ = "nifty_scalper_bot.core.app"


class _Mdm:
    _freshness_hardening_installed = True


class _Ws:
    _market_data_hardening_installed = True


class _DataHub:
    _synthetic_timestamp_guard_installed = True


def _native_store_quote(self):
    return None


def _native_canonicalize_tick_payload(self):
    return None


def _native_get_cached_ltp(self):
    return None


for _fn in (
    _native_store_quote,
    _native_canonicalize_tick_payload,
    _native_get_cached_ltp,
):
    _fn.__module__ = "nifty_scalper_bot.data.data_hub"


class _NativeDataHub:
    _synthetic_timestamp_guard_installed = True
    store_quote = _native_store_quote
    _canonicalize_tick_payload = _native_canonicalize_tick_payload
    get_cached_ltp = _native_get_cached_ltp


def _install_native_app_module(monkeypatch) -> None:
    import types

    app_mod = types.ModuleType("nifty_scalper_bot.core.app")
    app_mod._polling_failover_supervisor_iteration = _native_polling_supervisor
    app_mod._native_runtime_hardening_bootstrap = True
    monkeypatch.setitem(sys.modules, "nifty_scalper_bot.core.app", app_mod)


def test_runtime_install_proof_uses_context_instances(monkeypatch) -> None:
    monkeypatch.setattr(sys, "meta_path", [_PlainFinder()])
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=_Ws(),
        data_hub=_DataHub(),
    )

    proof = build_runtime_install_proof(ctx)

    assert proof["market_data_manager_hardened"] is True
    assert proof["websocket_hardened"] is True
    assert proof["datahub_synthetic_guard_installed"] is True
    assert proof["core_app_import_hook_required"] is False
    assert proof["core_app_import_hook_installed"] is False
    assert proof["datahub_import_hook_installed"] is False
    assert proof["import_hook_counts"] == {"core_app": 0, "datahub": 0}


def test_runtime_install_proof_rejects_legacy_core_import_hooks(monkeypatch) -> None:
    _install_native_app_module(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", [_LegacyCoreHook(), _LegacyCoreHook()])

    proof = build_runtime_install_proof(None)

    assert proof["core_app_native_bootstrap_loaded"] is True
    assert proof["core_app_import_hook_required"] is False
    assert proof["core_app_import_hook_installed"] is True
    assert proof["import_hook_counts"]["core_app"] == 2
    assert proof["all_required_installed"] is False


def test_runtime_install_proof_all_required_requires_every_marker(monkeypatch) -> None:
    _install_native_app_module(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", [_PlainFinder()])
    partial_ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=None,
        data_hub=_NativeDataHub(),
    )

    proof = build_runtime_install_proof(partial_ctx)

    assert proof["market_data_manager_hardened"] is True
    assert proof["websocket_hardened"] is False
    assert proof["core_app_native_bootstrap_loaded"] is True
    assert proof["all_required_installed"] is False


def test_runtime_install_proof_accepts_native_datahub_without_import_hook(
    monkeypatch,
) -> None:
    _install_native_app_module(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", [_PlainFinder()])
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=_Ws(),
        data_hub=_NativeDataHub(),
    )

    proof = build_runtime_install_proof(ctx)

    assert proof["datahub_import_hook_installed"] is False
    assert proof["datahub_import_hook_required"] is False
    assert proof["datahub_native_guard_loaded"] is True
    assert proof["datahub_hardening_satisfied"] is True
    assert proof["datahub_hardening_mode"] == "native"
    assert proof["polling_failover_native_owner"] is True
    assert proof["core_app_native_bootstrap_loaded"] is True
    assert proof["core_app_import_hook_installed"] is False
    assert proof["import_hook_counts"]["core_app"] == 0
    assert proof["all_required_installed"] is True


def test_runtime_install_proof_rejects_native_datahub_wrong_method_owner(
    monkeypatch,
) -> None:
    _install_native_app_module(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", [_PlainFinder()])
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=_Ws(),
        data_hub=_DataHub(),
    )

    proof = build_runtime_install_proof(ctx)

    assert proof["datahub_hardening_satisfied"] is False
    assert proof["datahub_hardening_mode"] == "invalid_native_ownership"
    assert proof["all_required_installed"] is False


def test_runtime_install_proof_unloaded_datahub_needs_no_import_hook(
    monkeypatch,
) -> None:
    _install_native_app_module(monkeypatch)
    monkeypatch.delitem(sys.modules, "nifty_scalper_bot.data.data_hub", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_PlainFinder()])

    proof = build_runtime_install_proof(None)

    assert proof["datahub_import_hook_required"] is False
    assert proof["datahub_import_hook_installed"] is False
    assert proof["import_hook_counts"]["datahub"] == 0
    assert proof["datahub_hardening_satisfied"] is True
    assert proof["datahub_hardening_mode"] == "native_not_loaded"
