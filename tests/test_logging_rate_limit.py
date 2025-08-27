import logging
from uuid import uuid4

from src.utils.logging_tools import RateLimitFilter


def _make_logger():
    name = f"rate_limit_{uuid4()}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    return logger, records


def test_filter_allows_different_messages():
    logger, records = _make_logger()
    logger.addFilter(RateLimitFilter(interval=60))
    logger.error("Err %s", 1)
    logger.error("Err %s", 2)
    assert records == ["Err 1", "Err 2"]


def test_filter_suppresses_duplicates():
    logger, records = _make_logger()
    logger.addFilter(RateLimitFilter(interval=60))
    logger.info("hello")
    logger.info("hello")
    assert records == ["hello"]
