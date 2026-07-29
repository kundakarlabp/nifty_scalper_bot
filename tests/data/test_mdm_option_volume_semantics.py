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


# ============ REST / pull transport volume contract ============
# Zerodha REST /quote returns `volume` / `volume_traded_today` as volume traded
# TODAY -- cumulative, not per-interval. _normalize_tick() previously mapped it
# straight to `volume` with no volume_cumulative / volume_delta /
# volume_delta_untrusted, so a cached REST quote reaching the runner looked
# like explicit interval volume (RUNNER_OPTION_VOLUME_CLAMPED volume=28574260).

_SYM = "NFO:NIFTY26JUN24000CE"


def test_rest_cumulative_volume_is_not_published_as_interval_volume() -> None:
    """A REST quote must be labelled cumulative, never interval."""
    mdm = _mdm()
    out = mdm._normalize_tick(
        _SYM,
        {"last_price": 45.0, "volume_traded_today": 28_574_260,
         "instrument_token": 1, "source": "rest"},
    )
    assert out is not None
    assert out.get("volume_cumulative") == 28_574_260.0
    # First observation establishes the baseline: delta 0, never the daily total.
    assert out.get("volume_delta") == 0.0
    assert out["volume_transition"]["state"] == "baseline_initialized"
    assert out.get("volume") != 28_574_260.0


def test_untagged_rest_volume_key_is_also_treated_as_cumulative() -> None:
    """Zerodha's `volume` on this transport is the daily total, not interval."""
    mdm = _mdm()
    raw = mdm._prepare_rest_tick(
        {"last_price": 45.0, "volume": 28_574_260, "instrument_token": 1},
        source="rest",
    )
    out = mdm._normalize_tick(_SYM, raw)
    assert out is not None
    assert out.get("volume_cumulative") == 28_574_260.0
    assert out.get("volume_delta") == 0.0
    assert out["volume_transition"]["state"] == "baseline_initialized"


def test_explicit_interval_delta_is_not_relabelled_as_cumulative() -> None:
    """REGRESSION TEST 1: a genuine interval delta must pass through intact."""
    mdm = _mdm()
    out = mdm._normalize_tick(
        _SYM,
        {
            "last_price": 45.0,
            "volume_delta": 1_250.0,
            "volume_cumulative": 28_574_260,
            "source": "ws",
        },
    )
    assert out is not None
    assert out.get("volume_delta") == 1_250.0
    assert out.get("volume") == 1_250.0
    # Cumulative is retained separately, not conflated with the delta.
    assert out.get("volume_cumulative") == 28_574_260.0
    assert out.get("volume_delta_untrusted") is not True


def test_effective_volume_delta_only_remains_interval_volume() -> None:
    """Compatibility: effective-only interval input must not become cumulative."""
    mdm = _mdm()
    prepared = mdm._prepare_rest_tick(
        {
            "last_price": 45.0,
            "volume": 28_574_260,
            "effective_volume_delta": 65.0,
            "instrument_token": 1,
        },
        source="rest",
    )
    out = mdm._normalize_tick(_SYM, prepared)

    assert out is not None
    assert out["volume"] == 65.0
    assert out["volume_delta"] == 65.0
    assert out["effective_volume_delta"] == 65.0
    assert out.get("volume_cumulative") != 28_574_260.0


def test_public_rest_ingress_publishes_incremental_delta() -> None:
    """REST ingress must preserve its derived delta through queue and storage."""
    mdm = _mdm()
    mdm._is_duplicate = lambda *_args, **_kwargs: False  # type: ignore[method-assign]

    def process_immediately(tick: dict) -> None:
        mdm._process_queued_tick(tick)

    mdm._enqueue_tick_threadsafe = process_immediately  # type: ignore[method-assign]

    mdm.ingest_rest_quote(
        _SYM,
        {
            "last_price": 45.0,
            "volume": 28_574_260,
            "instrument_token": 1,
            "timestamp": datetime(2026, 1, 1, 9, 15, 1, tzinfo=timezone.utc),
        },
    )
    first = mdm.get_latest_tick(_SYM)

    mdm.ingest_rest_quote(
        _SYM,
        {
            "last_price": 45.2,
            "volume": 28_574_325,
            "instrument_token": 1,
            "timestamp": datetime(2026, 1, 1, 9, 15, 2, tzinfo=timezone.utc),
        },
    )
    second = mdm.get_latest_tick(_SYM)

    assert first is not None
    assert first["volume_delta"] == 0.0
    assert second is not None
    assert second["volume"] == 65.0
    assert second["volume_delta"] == 65.0
    assert second["volume_cumulative"] == 28_574_325.0


def _rest(mdm: MarketDataManager, cumulative: int) -> dict:
    """Drive the real REST ingress -> normalize path."""
    raw = mdm._prepare_rest_tick(
        {"last_price": 45.0, "volume": cumulative, "instrument_token": 1},
        source="rest",
    )
    return mdm._normalize_tick(_SYM, raw) or {}


def test_rest_cumulative_produces_real_incremental_delta() -> None:
    """END-TO-END: first REST observation 0, next a true incremental delta.

    Publishing volume_cumulative alone is NOT enough: that key is diagnostic
    only and _normalise_tick_volume_delta() never consumes it, so the baseline
    would never advance and every REST tick would publish delta 0.
    """
    mdm = _mdm()

    first = _rest(mdm, 28_574_260)
    assert first["volume_delta"] == 0.0
    assert first["volume_transition"]["state"] == "baseline_initialized"
    assert first["volume_cumulative"] == 28_574_260.0

    second = _rest(mdm, 28_574_325)
    assert second["volume_delta"] == 65.0
    assert second["effective_volume_delta"] == 65.0
    assert second["volume_transition"]["state"] == "accepted"
    assert second["volume_cumulative"] == 28_574_325.0
    # REST must not remain permanently untrusted once a baseline exists.
    assert second.get("volume_delta_untrusted") is not True


def test_ws_rest_ws_transition_shares_one_baseline() -> None:
    """WS -> REST fallback -> WS must share ONE baseline, no double counting."""
    mdm = _mdm()

    ws1 = mdm._normalize_tick(
        _SYM, {"last_price": 45.0, "volume_traded_today": 28_574_000,
               "instrument_token": 1, "source": "ws"}
    ) or {}
    assert ws1["volume_delta"] == 0.0            # initial baseline

    rest = _rest(mdm, 28_574_260)
    assert rest["volume_delta"] == 260.0         # 260 since the WS baseline
    assert rest["volume_transition"]["state"] == "accepted"

    ws2 = mdm._normalize_tick(
        _SYM, {"last_price": 45.2, "volume_traded_today": 28_574_560,
               "instrument_token": 1, "source": "ws"}
    ) or {}
    assert ws2["volume_delta"] == 300.0          # 300 since the REST hop
    assert ws2["volume_transition"]["state"] == "accepted"

    # One shared baseline: increments sum to the total span, no double count.
    assert (
        ws1["volume_delta"] + rest["volume_delta"] + ws2["volume_delta"]
        == 28_574_560 - 28_574_000
    )
    # And no delta was ever the cumulative daily total.
    for tick in (ws1, rest, ws2):
        assert tick["volume_delta"] != tick["volume_cumulative"]


def test_out_of_order_rest_does_not_move_baseline() -> None:
    """A stale REST quote must not corrupt or rewind the baseline."""
    mdm = _mdm()
    _rest(mdm, 28_574_260)
    _rest(mdm, 28_574_325)

    stale = _rest(mdm, 28_574_100)
    assert stale["volume_transition"]["state"] != "accepted"
    assert stale.get("effective_volume_delta") in (0, 0.0, None)

    # Baseline intact: a later legitimate value still measures from 28_574_325.
    resumed = _rest(mdm, 28_574_400)
    assert resumed["volume_delta"] == 75.0
