"""Nifty scalper bot package."""

try:
    from nifty_scalper_bot.data.pipeline_overlap_guard import install_candle_store_overlap_guard

    install_candle_store_overlap_guard()
except Exception:
    pass

try:
    from nifty_scalper_bot.data.ohlc_capacity_contract import (
        install_mdm_ohlc_capacity_contract,
    )

    install_mdm_ohlc_capacity_contract()
except Exception:
    # Startup remains fail-closed through the existing history/readiness gates;
    # tests and install proof verify this adapter is present in production.
    pass

__all__: list[str] = []
