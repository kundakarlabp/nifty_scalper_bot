from __future__ import annotations

"""Tests for instrument loading helpers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.boot import load_instruments


def test_first_env_prefers_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSTRUMENTS__CSV", raising=False)
    monkeypatch.setenv("FALLBACK", "/tmp/fallback.csv")
    monkeypatch.setenv("INSTRUMENTS_CSV", "/tmp/primary.csv")

    value = load_instruments._first_env("INSTRUMENTS__CSV", "INSTRUMENTS_CSV", "FALLBACK")

    assert value == "/tmp/primary.csv"


def test_load_instrument_store_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Settings wins over environment lookup.
    settings = SimpleNamespace(INSTRUMENTS_CSV="/tmp/settings.csv")
    monkeypatch.setenv("INSTRUMENTS_CSV", "/tmp/env.csv")

    with patch.object(load_instruments.InstrumentStore, "from_csv", return_value="sentinel") as mock_from_csv:
        result = load_instruments.load_instrument_store_from_settings(settings)

    mock_from_csv.assert_called_once_with("/tmp/settings.csv")
    assert result == "sentinel"


def test_load_instrument_store_requires_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSTRUMENTS_CSV", raising=False)

    with pytest.raises(FileNotFoundError):
        load_instruments.load_instrument_store_from_settings(SimpleNamespace())
