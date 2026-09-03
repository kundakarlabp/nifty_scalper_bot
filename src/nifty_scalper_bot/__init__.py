"""Nifty scalper bot package."""

try:
    from nifty_scalper_bot.data.pipeline_overlap_guard import install_candle_store_overlap_guard

    install_candle_store_overlap_guard()
except Exception:
    pass

__all__: list[str] = []
