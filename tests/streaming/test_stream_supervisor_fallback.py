from nifty_scalper_bot.core.app import _polling_fallback_degraded


def test_spot_stale_only_does_not_activate_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=True,
            options_fresh=True,
        )
        is False
    )


def test_futures_stale_activates_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=False,
            options_fresh=True,
        )
        is True
    )


def test_options_stale_activates_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=True,
            options_fresh=False,
        )
        is True
    )


def test_ws_disconnected_activates_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=False,
            lagging=False,
            futures_fresh=True,
            options_fresh=True,
        )
        is True
    )


def test_lagging_event_loop_activates_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=True,
            futures_fresh=True,
            options_fresh=True,
        )
        is True
    )
