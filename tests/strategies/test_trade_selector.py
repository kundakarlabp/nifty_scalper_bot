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
    assert best.side == 'CE'


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


def test_no_recent_real_tick_is_penalty_when_soft_mode() -> None:
    selector = TradeCandidateSelector(require_real_ticks_last_60s=False)
    quality = selector.evaluate_data_quality(
        _snapshot('NFO:NIFTY23500CE', 23500, real_ticks_last_60s=0)
    )
    assert quality.allowed is True
    assert quality.score < 10.0


def test_no_recent_real_tick_is_block_when_strict_mode() -> None:
    selector = TradeCandidateSelector(require_real_ticks_last_60s=True)
    quality = selector.evaluate_data_quality(
        _snapshot('NFO:NIFTY23500CE', 23500, real_ticks_last_60s=0)
    )
    assert quality.allowed is False
    assert 'no_recent_real_tick' in quality.reasons


def test_selector_sets_pe_side_for_pe_bias() -> None:
    selector = TradeCandidateSelector()
    best = selector.select_best_candidate(
        underlying='NIFTY',
        direction_bias='PE',
        atm_strike=23500,
        snapshots=[_snapshot('NFO:NIFTY23500PE', 23500)],
    )
    assert best is not None
    assert best.side == 'PE'


def test_selector_allows_higher_nifty_premium_default_cap() -> None:
    selector = TradeCandidateSelector()
    best = selector.select_best_candidate(
        underlying='NIFTY',
        direction_bias='PE',
        atm_strike=24050,
        snapshots=[_snapshot('NFO:NIFTY26MAY24050PE', 24050, ltp=374.95, bid=374.5, ask=375.4)],
    )
    assert best is not None


def _candidate_snapshot(**overrides: object) -> dict[str, object]:
    payload = _snapshot('NFO:NIFTY23500CE', 23500, side='CE')
    payload.update(overrides)
    return payload


def test_selector_midday_pause_enabled_blocks_candidates(monkeypatch, caplog) -> None:
    from nifty_scalper_bot.strategies import trade_selector as trade_selector_mod

    monkeypatch.setattr(trade_selector_mod, 'expiry_theta_block', lambda: (False, 'not_expiry_day'))
    monkeypatch.setattr(
        trade_selector_mod,
        'midday_pause_block',
        lambda: (True, 'midday_pause_11:30-13:15_ist'),
    )
    selector = TradeCandidateSelector()

    with caplog.at_level('INFO', logger=trade_selector_mod.LOGGER.name):
        ranked = selector.select_ranked_candidates(
            direction_bias='CE',
            atm_strike=23500,
            snapshots=[_candidate_snapshot()],
        )

    assert ranked == []
    assert any(
        record.__dict__.get('event') == 'CANDIDATE_SELECTION_BLOCKED'
        and 'midday_pause' in str(record.__dict__.get('reason'))
        for record in caplog.records
    )


def test_selector_midday_pause_disabled_continues_to_quality_filtering(monkeypatch, caplog) -> None:
    from nifty_scalper_bot.strategies import trade_selector as trade_selector_mod

    monkeypatch.setattr(trade_selector_mod, 'expiry_theta_block', lambda: (False, 'not_expiry_day'))
    monkeypatch.setattr(trade_selector_mod, 'midday_pause_block', lambda: (False, 'pause_disabled'))
    selector = TradeCandidateSelector()

    with caplog.at_level('INFO', logger=trade_selector_mod.LOGGER.name):
        ranked = selector.select_ranked_candidates(
            direction_bias='CE',
            atm_strike=23500,
            snapshots=[_candidate_snapshot()],
        )

    assert ranked
    assert ranked[0].symbol == 'NFO:NIFTY23500CE'
    assert not any(
        record.__dict__.get('event') == 'CANDIDATE_SELECTION_BLOCKED'
        and 'midday_pause' in str(record.__dict__.get('reason'))
        for record in caplog.records
    )
