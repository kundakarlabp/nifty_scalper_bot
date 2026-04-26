from __future__ import annotations

import logging

from nifty_scalper_bot.utils.logging import log_throttled


def test_log_throttled_accepts_extra_and_exc_info(caplog) -> None:
    logger = logging.getLogger('tests.log_throttled')
    with caplog.at_level(logging.ERROR):
        log_throttled(
            logger,
            key='test_log_throttle_exc',
            msg='throttle message',
            interval_sec=0.0,
            level=logging.ERROR,
            extra={'event': 'throttle_test', 'trace_id': 'abc'},
            exc_info=False,
        )
    assert any('throttle message' in record.message for record in caplog.records)
