"""Regression tests for native polling supervisor installation identity."""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.polling_failover_runtime import apply_app_patch


def test_apply_app_patch_replaces_unversioned_supervisor():
    async def legacy_supervisor(*args, **kwargs):
        return None, None

    module = SimpleNamespace(
        _polling_failover_supervisor_iteration=legacy_supervisor,
        _polling_fallback_degraded=lambda **kwargs: False,
    )

    assert apply_app_patch(module) is True
    installed = module._polling_failover_supervisor_iteration
    assert installed is not legacy_supervisor
    assert getattr(installed, "_nifty_polling_supervisor_version", None) == 2


def test_apply_app_patch_is_idempotent_for_versioned_supervisor():
    async def installed_supervisor(*args, **kwargs):
        return None, None

    setattr(installed_supervisor, "_nifty_polling_supervisor_version", 2)
    module = SimpleNamespace(
        _polling_failover_supervisor_iteration=installed_supervisor,
        _polling_fallback_degraded=lambda **kwargs: False,
    )

    assert apply_app_patch(module) is False
    assert module._polling_failover_supervisor_iteration is installed_supervisor
