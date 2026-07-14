from __future__ import annotations

import sys
from types import SimpleNamespace

from nifty_scalper_bot.core.runtime_install_proof import build_runtime_install_proof


class _CoreHook:
    _nifty_scalper_core_app_patch_hook = True


class _DataHubHook:
    _nifty_scalper_datahub_synthetic_guard_hook = True


class _PlainFinder:
    pass


class _Mdm:
    _freshness_hardening_installed = True


class _Ws:
    _market_data_hardening_installed = True


class _DataHub:
    _synthetic_timestamp_guard_installed = True


def test_runtime_install_proof_uses_context_instances(monkeypatch) -> None:
    monkeypatch.setattr(sys, "meta_path", [_CoreHook(), _DataHubHook(), _PlainFinder()])
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=_Ws(),
        data_hub=_DataHub(),
    )

    proof = build_runtime_install_proof(ctx)

    assert proof["market_data_manager_hardened"] is True
    assert proof["websocket_hardened"] is True
    assert proof["datahub_synthetic_guard_installed"] is True
    assert proof["core_app_import_hook_installed"] is True
    assert proof["datahub_import_hook_installed"] is True
    assert proof["import_hook_counts"] == {"core_app": 1, "datahub": 1}


def test_runtime_install_proof_reports_duplicate_import_hooks(monkeypatch) -> None:
    monkeypatch.setattr(sys, "meta_path", [_CoreHook(), _CoreHook(), _DataHubHook()])

    proof = build_runtime_install_proof(None)

    assert proof["core_app_import_hook_installed"] is False
    assert proof["datahub_import_hook_installed"] is True
    assert proof["import_hook_counts"]["core_app"] == 2


def test_runtime_install_proof_all_required_requires_every_marker(monkeypatch) -> None:
    monkeypatch.setattr(sys, "meta_path", [_CoreHook(), _DataHubHook()])
    partial_ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=None,
        data_hub=_DataHub(),
    )

    proof = build_runtime_install_proof(partial_ctx)

    assert proof["market_data_manager_hardened"] is True
    assert proof["websocket_hardened"] is False
    assert proof["all_required_installed"] is False


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


def test_runtime_install_proof_accepts_native_datahub_without_import_hook(
    monkeypatch,
) -> None:
    import types

    app_mod = types.ModuleType("nifty_scalper_bot.core.app")
    app_mod._polling_failover_runtime_patch_installed = True
    monkeypatch.setitem(sys.modules, "nifty_scalper_bot.core.app", app_mod)
    monkeypatch.setattr(sys, "meta_path", [_CoreHook()])
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
    assert proof["all_required_installed"] is True


def test_runtime_install_proof_rejects_native_datahub_wrong_method_owner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "meta_path", [_CoreHook()])
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        websocket_manager=_Ws(),
        data_hub=_DataHub(),
    )

    proof = build_runtime_install_proof(ctx)

    assert proof["datahub_hardening_satisfied"] is False
    assert proof["datahub_hardening_mode"] == "invalid_native_ownership"
