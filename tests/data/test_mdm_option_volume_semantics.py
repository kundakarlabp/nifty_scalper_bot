from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _mdm() -> MarketDataManager:
    mdm = MarketDataManager(kite=None)
    sym = "NFO:NIFTY26JUN24000CE"
    mdm._symbol_by_token[1] = sym
    mdm._symbol_to_token[sym] = 1
    mdm._token_by_symbol[sym] = 1
    mdm._desired_tokens.add(1)
    return mdm


def _raw(cumulative: int, *, token: int = 1, ts: int = 1) -> dict:
    return {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "instrument_token": token,
        "volume_traded_today": cumulative,
        "last_price": 100,
        "timestamp": datetime(2026, 1, 1, 9, 15, ts, tzinfo=timezone.utc),
    }


def test_option_volume_baseline_normal_delta_generation_reset_and_duplicate() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    first = mdm._normalise_tick_volume_delta(symbol, _raw(1_000))
    second = mdm._normalise_tick_volume_delta(symbol, _raw(1_025, ts=2))
    duplicate = mdm._normalise_tick_volume_delta(symbol, _raw(1_025, ts=3))
    mdm._subscription_generation += 1
    mdm._symbol_subscription_generation[symbol] = mdm._subscription_generation
    reset = mdm._normalise_tick_volume_delta(symbol, _raw(5_000, ts=4))

    assert first["volume_transition"]["state"] == "baseline_initialized"
    assert first["volume_delta"] == 0
    assert second["volume_transition"]["state"] == "accepted"
    assert second["effective_volume_delta"] == 25
    assert duplicate["volume_transition"]["state"] == "duplicate"
    assert duplicate["volume_delta"] == 0
    assert reset["volume_transition"]["state"] == "baseline_initialized"
    assert reset["volume_delta"] == 0


def test_option_volume_rollback_and_suspicious_jump_do_not_poison_candle_volume() -> (
    None
):
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._normalise_tick_volume_delta(symbol, _raw(1_000))
    rollback = mdm._normalise_tick_volume_delta(symbol, _raw(900, ts=2))
    suspicious = mdm._normalise_tick_volume_delta(symbol, _raw(2_000_000, ts=3))
    after = mdm._normalise_tick_volume_delta(symbol, _raw(1_050, ts=4))

    assert rollback["volume_transition"]["state"] == "counter_rollback"
    assert rollback["volume_delta"] == 0
    assert suspicious["volume_transition"]["state"] == "suspicious_jump"
    assert suspicious["volume_delta_untrusted"] is True
    assert suspicious["volume"] == 0
    assert suspicious["volume_cumulative"] == 2_000_000
    assert after["volume_delta"] == 50
    assert after["volume"] == 50


def test_raw_cumulative_never_becomes_completed_candle_volume() -> None:
    mdm = _mdm()
    processed = [
        mdm._normalise_tick_volume_delta("NFO:NIFTY26JUN24000CE", _raw(748_735_130)),
        mdm._normalise_tick_volume_delta(
            "NFO:NIFTY26JUN24000CE", _raw(748_735_155, ts=2)
        ),
    ]
    assert processed[0]["volume"] == 0
    assert processed[1]["volume"] == 25
    assert processed[1]["volume_cumulative"] == 748_735_155
    assert processed[1]["volume"] != processed[1]["volume_cumulative"]


def test_option_volume_out_of_order_tick_does_not_reset_baseline() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._normalise_tick_volume_delta(symbol, _raw(1_000, ts=10))

    late = mdm._normalise_tick_volume_delta(symbol, _raw(900, ts=5))
    after = mdm._normalise_tick_volume_delta(symbol, _raw(1_020, ts=11))

    assert late["volume_transition"]["state"] == "out_of_order"
    assert late["volume_delta"] == 0
    assert late["volume_delta_untrusted"] is True
    assert after["volume_transition"]["state"] == "accepted"
    assert after["volume_delta"] == 20


def test_counter_reset_requires_explicit_or_session_evidence() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._normalise_tick_volume_delta(symbol, _raw(10_000, ts=1))

    rollback = mdm._normalise_tick_volume_delta(symbol, _raw(100, ts=2))
    after_rollback = mdm._normalise_tick_volume_delta(symbol, _raw(10_025, ts=3))
    explicit_reset_raw = _raw(50, ts=4)
    explicit_reset_raw["counter_reset"] = True
    reset = mdm._normalise_tick_volume_delta(symbol, explicit_reset_raw)
    after_reset = mdm._normalise_tick_volume_delta(symbol, _raw(75, ts=5))

    assert rollback["volume_transition"]["state"] == "counter_rollback"
    assert rollback["volume_delta"] == 0
    assert after_rollback["volume_delta"] == 25
    assert reset["volume_transition"]["state"] == "counter_reset"
    assert reset["volume_delta"] == 0
    assert after_reset["volume_delta"] == 25
