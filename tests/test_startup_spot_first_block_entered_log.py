import logging

from nifty_scalper_bot.core.app import _safe_startup_log


def test_startup_spot_first_block_entered_log(caplog) -> None:
    logger = logging.getLogger('test.startup.block')
    with caplog.at_level(logging.INFO):
        _safe_startup_log(logger, logging.INFO, 'STARTUP_WS_SPOT_FIRST_DECISION', 'STARTUP_WS_SPOT_FIRST_DECISION')
        logger.info('STARTUP_SPOT_FIRST_BLOCK_ENTERED websocket_enabled=True mdm_id=1')
    text = caplog.text
    assert 'STARTUP_WS_SPOT_FIRST_DECISION' in text
    assert 'STARTUP_SPOT_FIRST_BLOCK_ENTERED' in text
    assert text.index('STARTUP_WS_SPOT_FIRST_DECISION') < text.index('STARTUP_SPOT_FIRST_BLOCK_ENTERED')
