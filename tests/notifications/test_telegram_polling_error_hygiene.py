from __future__ import annotations

import logging


def test_ptb_default_polling_error_filter_suppresses_generic_line() -> None:
    from nifty_scalper_bot.notifications import _TelegramUpdaterDefaultErrorFilter

    filt = _TelegramUpdaterDefaultErrorFilter()
    record = logging.LogRecord(
        name="telegram.ext.Updater",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Exception happened while polling for updates.",
        args=(),
        exc_info=None,
        func="default_error_callback",
    )

    assert filt.filter(record) is False


def test_ptb_default_polling_error_filter_preserves_other_logs() -> None:
    from nifty_scalper_bot.notifications import _TelegramUpdaterDefaultErrorFilter

    filt = _TelegramUpdaterDefaultErrorFilter()
    record = logging.LogRecord(
        name="telegram.ext.Updater",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="TELEGRAM_POLLING_CONFLICT another process is polling",
        args=(),
        exc_info=None,
        func="custom_error_callback",
    )

    assert filt.filter(record) is True


def test_host_wide_polling_owner_blocks_same_token_duplicate(tmp_path, monkeypatch) -> None:
    from nifty_scalper_bot.notifications.telegram_runtime_registry import (
        claim_polling_owner,
        release_polling_owner,
    )

    monkeypatch.setenv("NIFTY_RUNTIME_LOCK_DIR", str(tmp_path))
    token = "123:abc"

    assert claim_polling_owner(token=token, owner="owner-a") is True
    try:
        assert claim_polling_owner(token=token, owner="owner-b") is False
    finally:
        release_polling_owner(token=token, owner="owner-a")

    assert claim_polling_owner(token=token, owner="owner-b") is True
    release_polling_owner(token=token, owner="owner-b")
