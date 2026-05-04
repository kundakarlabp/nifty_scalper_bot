import logging
from nifty_scalper_bot.core.app import _safe_startup_log


def test_safe_startup_log_handles_expected_args(caplog):
    logger = logging.getLogger("test.startup.log")
    with caplog.at_level(logging.INFO):
        _safe_startup_log(
            logger,
            logging.INFO,
            "STARTUP_SPOT_TOKEN_REQUEST_DONE",
            "STARTUP_SPOT_TOKEN_REQUEST_DONE symbol=%s token=%d subscribed=%s desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
            "NSE:NIFTY",
            256265,
            True,
            5,
            2,
            True,
        )
    assert "STARTUP_SPOT_TOKEN_REQUEST_DONE" in caplog.text
    assert "STARTUP_LOG_FORMAT_ERROR" not in caplog.text
