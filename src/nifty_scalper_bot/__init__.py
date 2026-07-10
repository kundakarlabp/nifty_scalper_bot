"""Nifty scalper bot package."""

try:
    from nifty_scalper_bot.data.pipeline_overlap_guard import install_candle_store_overlap_guard

    install_candle_store_overlap_guard()
except Exception:
    pass

try:
    from nifty_scalper_bot.runtime_live_safety_hotfixes import install_live_safety_hotfixes

    install_live_safety_hotfixes()
except Exception:
    pass

__all__: list[str] = []
