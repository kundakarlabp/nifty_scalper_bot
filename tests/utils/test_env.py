import os

import pytest

from nifty_scalper_bot.utils import env


def test_get_bool_parses_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOL_KEY", "yes")
    assert env.get_bool("BOOL_KEY") is True
    monkeypatch.setenv("BOOL_KEY", "off")
    assert env.get_bool("BOOL_KEY", default=True) is False
    monkeypatch.setenv("BOOL_KEY", "unknown")
    assert env.get_bool("BOOL_KEY", default=True) is True


def test_get_int_and_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INT_KEY", "42")
    monkeypatch.setenv("FLOAT_KEY", "3.14")
    assert env.get_int("INT_KEY") == 42
    assert env.get_float("FLOAT_KEY") == pytest.approx(3.14)
    monkeypatch.setenv("INT_KEY", "invalid")
    assert env.get_int("INT_KEY", default=7) == 7
    monkeypatch.setenv("FLOAT_KEY", "not-a-number")
    assert env.get_float("FLOAT_KEY", default=1.5) == pytest.approx(1.5)


def test_get_csv_and_coalesce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSV_KEY", '"alpha", "beta", gamma')
    assert env.get_csv("CSV_KEY") == ["alpha", "beta", "gamma"]
    monkeypatch.setenv("CSV_KEY", '"NIFTY"')
    assert env.get_csv("CSV_KEY") == ["NIFTY"]
    monkeypatch.setenv("CSV_KEY", '"NIFTY"\nPOLLING__INTERVAL_SEC=0.70')
    assert env.get_csv("CSV_KEY") == ["NIFTY"]
    monkeypatch.delenv("CSV_KEY", raising=False)
    assert env.get_csv("CSV_KEY") == []

    monkeypatch.setenv("PRIMARY_BOOL", "true")
    assert env.coalesce_bool("PRIMARY_BOOL", "SECONDARY_BOOL") is True
    monkeypatch.delenv("PRIMARY_BOOL", raising=False)
    monkeypatch.setenv("SECONDARY_BOOL", "1")
    assert env.coalesce_bool("PRIMARY_BOOL", "SECONDARY_BOOL") is True
    monkeypatch.setenv("PRIMARY_BOOL", "unknown")
    assert env.coalesce_bool("PRIMARY_BOOL", "SECONDARY_BOOL") is False
    assert env.coalesce_bool("PRIMARY_BOOL", "SECONDARY_BOOL", default=True) is True

    monkeypatch.delenv("SECONDARY_BOOL", raising=False)
    monkeypatch.setenv("PRIMARY_INT", "-5")
    monkeypatch.setenv("SECONDARY_INT", "10")
    assert env.coalesce_int("PRIMARY_INT", "SECONDARY_INT", default=3) == -5
    monkeypatch.delenv("PRIMARY_INT", raising=False)
    assert env.coalesce_int("PRIMARY_INT", "SECONDARY_INT", default=3) == 10

    monkeypatch.delenv("SECONDARY_INT", raising=False)
    assert env.coalesce_float("A", "B", default=2.5) == pytest.approx(2.5)
    monkeypatch.setenv("B", "5.5")
    assert env.coalesce_float("A", "B", default=1.0) == pytest.approx(5.5)


def test_coalesce_str_prefers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRST", "value1")
    monkeypatch.setenv("SECOND", "value2")
    assert env.coalesce_str("FIRST", "SECOND", default="fallback") == "value1"
    monkeypatch.delenv("FIRST", raising=False)
    assert env.coalesce_str("FIRST", "SECOND", default="fallback") == "value2"
    monkeypatch.delenv("SECOND", raising=False)
    assert env.coalesce_str("FIRST", "SECOND", default="fallback") == "fallback"


@pytest.fixture(autouse=True)
def _restore_env() -> None:
    yield
    os.environ.pop("BOOL_KEY", None)
    os.environ.pop("INT_KEY", None)
    os.environ.pop("FLOAT_KEY", None)
    os.environ.pop("CSV_KEY", None)
    os.environ.pop("PRIMARY_BOOL", None)
    os.environ.pop("SECONDARY_BOOL", None)
    os.environ.pop("PRIMARY_INT", None)
    os.environ.pop("SECONDARY_INT", None)
    os.environ.pop("A", None)
    os.environ.pop("B", None)
    os.environ.pop("FIRST", None)
    os.environ.pop("SECOND", None)
