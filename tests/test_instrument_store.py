"""Tests for :mod:`src.broker.instruments`."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.broker.instruments import InstrumentStore


def test_from_csv_validation(tmp_path: Path) -> None:
    """Invalid rows should raise ``ValueError``."""
    p = tmp_path / "bad.csv"
    p.write_text("token,symbol,lot_size,tick_size\n,ABC,25,0.05\n")
    with pytest.raises(ValueError):
        InstrumentStore.from_csv(str(p))


def test_reload_logs_diffs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Reloading should update the store and log differences."""
    csv = tmp_path / "inst.csv"
    csv.write_text("token,symbol,lot_size,tick_size\n111,AAA,1,0.05\n")
    store = InstrumentStore.from_csv(str(csv))
    csv.write_text("token,symbol,lot_size,tick_size\n111,AAA,1,0.05\n222,BBB,2,0.05\n")
    with caplog.at_level(logging.INFO):
        store.reload()
    assert store.by_token(222) is not None
    assert "added=[222]" in caplog.text

