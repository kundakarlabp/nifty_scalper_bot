"""Regression tests for day timeframe aliases."""

from __future__ import annotations

import logging

from src.config import settings
from src.data import source


def test_coerce_interval_day_aliases(monkeypatch, caplog) -> None:
    """Ensure daily aliases map to the day interval without warnings."""

    monkeypatch.setattr(settings.data, "timeframe", "day", raising=False)

    aliases = ("day", "1day", "1d", "daily")
    for alias in aliases:
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            assert source._coerce_interval(alias) == "day"
            assert not caplog.records
