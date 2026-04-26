from __future__ import annotations

from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


def _snapshot(symbol: str, strike: int, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'symbol': symbol,
        'strike': strike,
        'ltp': 120.0,
        'bid': 118.0,
        'ask': 122.0,
        'mid': 120.0,
        'tick_age_s': 0.8,
        'ohlc_valid': True,
        'real_ticks_last_60s': 5,
        'latest_candle_provisional': False,
        'latest_candle_synthetic': False,
    }
    payload.update(overrides)
    return payload


def test_selector_picks_highest_valid_candidate() -> None:
    selector = TradeCandidateSelector()
    best = selector.select_best_candidate(
        underlying='NIFTY',
        direction_bias='CE',
        atm_strike=23500,
        snapshots=[
            _snapshot('NFO:NIFTY23500CE', 23500),
            _snapshot('NFO:NIFTY23450CE', 23450, bid=116.0, ask=124.0, mid=120.0),
            _snapshot('NFO:NIFTY23650CE', 23650),  # far strike => rejected
            _snapshot('NFO:NIFTY23500PE', 23500),  # direction mismatch => rejected
        ],
    )
    assert best is not None
    assert best.symbol == 'NFO:NIFTY23500CE'


def test_selector_rejects_wide_spread_invalid_bid_ask_and_stale() -> None:
    selector = TradeCandidateSelector(max_tick_age_s=2.0, max_option_spread_pct=0.05)
    assert (
        selector.select_best_candidate(
            underlying='NIFTY',
            direction_bias='PE',
            atm_strike=23500,
            snapshots=[_snapshot('NFO:NIFTY23500PE', 23500, bid=0.0, ask=10.0, mid=5.0)],
        )
        is None
    )
    assert (
        selector.select_best_candidate(
            underlying='NIFTY',
            direction_bias='PE',
            atm_strike=23500,
            snapshots=[_snapshot('NFO:NIFTY23500PE', 23500, bid=100.0, ask=120.0, mid=110.0)],
        )
        is None
    )
    assert (
        selector.select_best_candidate(
            underlying='NIFTY',
            direction_bias='PE',
            atm_strike=23500,
            snapshots=[_snapshot('NFO:NIFTY23500PE', 23500, tick_age_s=9.0)],
        )
        is None
    )


def test_selector_default_tick_age_from_env(monkeypatch) -> None:
    monkeypatch.delenv('MAX_OPTION_TICK_AGE_SECONDS', raising=False)
    selector = TradeCandidateSelector()
    assert selector.max_tick_age_s == 10.0
