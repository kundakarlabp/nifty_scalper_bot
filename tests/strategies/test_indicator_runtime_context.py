from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def test_indicator_runtime_context_is_merged_without_overwrite() -> None:
    engine = IndicatorEngine()
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for idx in range(25):
        engine.update_price(
            'NFO:NIFTY26MAY24200CE',
            {'open': 100 + idx, 'high': 101 + idx, 'low': 99 + idx, 'close': 100 + idx},
            volume=10,
            timestamp=base + timedelta(minutes=idx),
        )
    engine.set_indicators(
        'NFO:NIFTY26MAY24200CE',
        {'atm_strike': 24200, 'is_selected_option': True},
    )
    indicators = engine.get_indicators('NFO:NIFTY26MAY24200CE')
    assert indicators['atm_strike'] == 24200
    assert indicators['is_selected_option'] is True
    assert indicators['close'] == 124.0

def test_indicator_runtime_context_preserves_temporal_orderflow_fields() -> None:
    engine = IndicatorEngine()
    symbol = "NFO:NIFTY26SEP25000CE"
    engine.set_runtime_context(
        symbol,
        {
            "ofi_ready": True,
            "ofi_event": 25.0,
            "ofi_1s": 80.0,
            "ofi_3s": 120.0,
            "ofi_1s_normalized": 0.32,
            "ofi_3s_normalized": 0.24,
            "ofi_update_count_1s": 4,
            "ofi_update_count_3s": 9,
            "ofi_source": "ws_full_depth",
            "queue_imbalance_top": 0.15,
        },
    )

    context = engine.get_runtime_context(symbol)

    assert context["ofi_ready"] is True
    assert context["ofi_1s_normalized"] == 0.32
    assert context["ofi_update_count_1s"] == 4
    assert context["ofi_source"] == "ws_full_depth"
    assert context["queue_imbalance_top"] == 0.15

