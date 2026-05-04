import logging
from nifty_scalper_bot.core.app import _safe_startup_log


def test_startup_spot_first_cluster_logging(caplog):
    logger = logging.getLogger('test.startup.cluster')
    events = [
        'STARTUP_WS_SPOT_FIRST_DECISION','STARTUP_SPOT_REGISTER_BEGIN','STARTUP_SPOT_REGISTER_DONE',
        'STARTUP_SPOT_TOKEN_REQUEST_BEGIN','STARTUP_SPOT_TOKEN_REQUEST_DONE','STARTUP_WS_START_BEGIN',
        'STARTUP_WS_START_DONE','STARTUP_WS_CONNECT_TASK_CREATED',
    ]
    with caplog.at_level(logging.INFO):
        for ev in events:
            _safe_startup_log(logger, logging.INFO, ev, f"{ev} symbol=%s", "NSE:NIFTY")
    for ev in events:
        assert ev in caplog.text
