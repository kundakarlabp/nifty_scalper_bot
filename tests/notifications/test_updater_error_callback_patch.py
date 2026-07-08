from __future__ import annotations

from telegram.error import TimedOut

from nifty_scalper_bot.notifications.updater_error_callback_patch import (
    classify_updater_polling_error,
    ensure_polling_error_callback,
)


def test_polling_kwargs_receive_callback_when_missing() -> None:
    kwargs: dict[str, object] = {}

    result = ensure_polling_error_callback(kwargs)

    assert result is kwargs
    assert callable(kwargs["error_callback"])


def test_polling_kwargs_preserve_existing_callback() -> None:
    def custom_callback(exc: object) -> None:
        return None

    kwargs: dict[str, object] = {"error_callback": custom_callback}

    ensure_polling_error_callback(kwargs)

    assert kwargs["error_callback"] is custom_callback


def test_polling_timeout_classified_as_transient() -> None:
    assert classify_updater_polling_error(TimedOut("read timeout")) == "transient"


def test_polling_conflict_classified() -> None:
    assert classify_updater_polling_error(RuntimeError("Conflict: terminated by other getUpdates request")) == "conflict"
