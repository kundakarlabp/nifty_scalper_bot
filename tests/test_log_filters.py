import logging
from uuid import uuid4

import src.utils.log_filters as lf


def _make_logger() -> tuple[logging.Logger, list[str]]:
    name = f"dedup_{uuid4()}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    return logger, records


def test_filter_suppresses_duplicates(monkeypatch) -> None:
    logger, records = _make_logger()
    flt = lf.DedupFilter([r"foo"], window_s=60)
    t = 0.0
    monkeypatch.setattr(lf.time, "time", lambda: t)
    logger.addFilter(flt)
    logger.warning("foo")
    logger.warning("foo")
    assert records == ["foo"]


def test_filter_allows_after_window(monkeypatch) -> None:
    logger, records = _make_logger()
    flt = lf.DedupFilter([r"foo"], window_s=60)
    t = 0.0
    monkeypatch.setattr(lf.time, "time", lambda: t)
    logger.addFilter(flt)
    logger.warning("foo")
    t = 61.0
    logger.warning("foo")
    assert records == ["foo", "foo"]
