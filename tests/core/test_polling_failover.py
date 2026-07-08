from __future__ import annotations

from nifty_scalper_bot.core.polling_failover import decide_polling_fallback


def test_options_stale_flag_but_selected_age_below_threshold_does_not_activate() -> None:
    decision = decide_polling_fallback(
        ws_ok=True,
        lagging=False,
        futures_fresh=True,
        options_fresh=False,
        quote_stale_ms=120_000,
        feed_health={
            "selected_ce_age_ms": 70,
            "selected_pe_age_ms": 68,
            "options_age_ms": 70,
        },
        data_age_ms=70,
    )

    assert decision.activate is False
    assert decision.reason is None
    assert decision.max_age_ms == 70
    assert decision.threshold_ms == 120_000


def test_options_stale_flag_activates_when_selected_age_crosses_threshold() -> None:
    decision = decide_polling_fallback(
        ws_ok=True,
        lagging=False,
        futures_fresh=True,
        options_fresh=False,
        quote_stale_ms=120_000,
        feed_health={
            "selected_ce_age_ms": 119_999,
            "selected_pe_age_ms": 120_000,
        },
        data_age_ms=70,
    )

    assert decision.activate is True
    assert decision.reason == "options_stale"
    assert decision.max_age_ms == 120_000


def test_websocket_unhealthy_still_activates_even_when_age_is_low() -> None:
    decision = decide_polling_fallback(
        ws_ok=False,
        lagging=False,
        futures_fresh=True,
        options_fresh=True,
        quote_stale_ms=120_000,
        feed_health={"options_age_ms": 70},
        data_age_ms=70,
    )

    assert decision.activate is True
    assert decision.reason == "websocket_unhealthy"


def test_futures_stale_requires_age_threshold_when_websocket_is_healthy() -> None:
    fresh_enough = decide_polling_fallback(
        ws_ok=True,
        lagging=False,
        futures_fresh=False,
        options_fresh=True,
        quote_stale_ms=120_000,
        feed_health={"futures_age_ms": 70},
        data_age_ms=70,
    )
    stale = decide_polling_fallback(
        ws_ok=True,
        lagging=False,
        futures_fresh=False,
        options_fresh=True,
        quote_stale_ms=120_000,
        feed_health={"futures_age_ms": 120_000},
        data_age_ms=70,
    )

    assert fresh_enough.activate is False
    assert stale.activate is True
    assert stale.reason == "futures_stale"
