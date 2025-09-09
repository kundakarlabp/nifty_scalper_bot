"""Additional coverage tests for simple helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import importlib
import pytest

from src.boot.load_instruments import load_instrument_store_from_settings
from src.data.base_source import BaseDataSource
from src.features.range import range_score


def test_load_instrument_store_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        load_instrument_store_from_settings(SimpleNamespace())


def test_load_instrument_store_valid_path() -> None:
    sample = Path("data/instruments_sample.csv")
    store = load_instrument_store_from_settings(SimpleNamespace(INSTRUMENTS_CSV=str(sample)))
    assert store is not None


def test_range_score_bounds() -> None:
    feats = SimpleNamespace(mom_norm=0.0, atr_pct=0.05)
    assert 0.9 <= range_score(feats) <= 1.0


def test_import_interfaces() -> None:
    assert importlib.import_module("src.interfaces") is not None


class _DummySource(BaseDataSource):
    _last_tick_ts = datetime(2024, 1, 1)
    _last_bar_open_ts = datetime(2024, 1, 1)
    _tf_seconds = 120


def test_base_data_source_accessors() -> None:
    src = _DummySource()
    assert src.last_tick_ts() == datetime(2024, 1, 1)
    assert src.last_bar_open_ts() == datetime(2024, 1, 1)
    assert src.timeframe_seconds == 120
