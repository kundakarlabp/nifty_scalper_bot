from nifty_scalper_bot.strategies.signal_generator import Signal


def _signal(symbol: str, **metadata):
    return Signal(
        action="BUY",
        symbol=symbol,
        quantity=75,
        confidence=0.9,
        reason="setup",
        stop_loss=90.0,
        take_profit=120.0,
        metadata={"strategy": "VWAPPro", **metadata},
    )


def test_same_setup_is_stable_across_strike_rotation():
    first = _signal(
        "NFO:NIFTY2680624500CE",
        setup_candle_timestamp="2026-07-31T09:22:00+05:30",
    )
    rotated = _signal(
        "NFO:NIFTY2680624550CE",
        setup_candle_timestamp="2026-07-31T09:22:00+05:30",
    )

    assert first.deterministic_id == rotated.deterministic_id


def test_setup_side_remains_part_of_identity():
    ce = _signal(
        "NFO:NIFTY2680624500CE",
        setup_candle_timestamp="2026-07-31T09:22:00+05:30",
    )
    pe = _signal(
        "NFO:NIFTY2680624500PE",
        setup_candle_timestamp="2026-07-31T09:22:00+05:30",
    )

    assert ce.deterministic_id != pe.deterministic_id


def test_new_setup_candle_gets_new_identity():
    first = _signal(
        "NFO:NIFTY2680624500CE",
        setup_candle_timestamp="2026-07-31T09:22:00+05:30",
    )
    next_setup = _signal(
        "NFO:NIFTY2680624500CE",
        setup_candle_timestamp="2026-07-31T09:23:00+05:30",
    )

    assert first.deterministic_id != next_setup.deterministic_id


def test_explicit_setup_id_has_priority_over_tick_timestamp():
    first = _signal(
        "NFO:NIFTY2680624500CE",
        setup_id="vwap-reclaim-17",
        timestamp="2026-07-31T09:22:01+05:30",
    )
    retry = _signal(
        "NFO:NIFTY2680624550CE",
        setup_id="vwap-reclaim-17",
        timestamp="2026-07-31T09:24:59+05:30",
    )

    assert first.deterministic_id == retry.deterministic_id


def test_strategy_remains_part_of_identity():
    vwap = _signal(
        "NFO:NIFTY2680624500CE",
        setup_candle_timestamp="2026-07-31T09:22:00+05:30",
    )
    smc = vwap.with_metadata(strategy="SMCLiquidity")

    assert vwap.deterministic_id != smc.deterministic_id
