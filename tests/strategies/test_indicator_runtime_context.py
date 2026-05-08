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
