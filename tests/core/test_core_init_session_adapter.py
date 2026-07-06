from __future__ import annotations


def test_core_exports_nifty_scalper_app_with_session_adapter() -> None:
    import nifty_scalper_bot.core as core
    import nifty_scalper_bot.core.app as app_module

    assert core.NiftyScalperApp is app_module.NiftyScalperApp
    assert getattr(app_module.compute_live_readiness, "_session_readiness_adapted", False) is True
