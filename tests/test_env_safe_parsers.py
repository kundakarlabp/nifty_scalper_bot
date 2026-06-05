"""Tests for safe env/config numeric parsers (inline-comment tolerant)."""
import pytest
from nifty_scalper_bot.config.env_utils import (
    parse_float_env, parse_int_env, parse_bool_env,
)


@pytest.mark.parametrize("val,exp", [
    ("30.0", 30.0),
    ("30.0     # Seconds between signals", 30.0),     # the crash case
    ("30.0     # Seconds between signals on same underlying", 30.0),
    (" 30.0 ", 30.0),
    ('"30.0"', 30.0),
    ("", 99.0),          # blank -> default
    (None, 99.0),        # None -> default
    ("bad_value", 99.0), # invalid -> default
    (30, 30.0),          # int passthrough
    (30.5, 30.5),        # float passthrough
])
def test_parse_float_env(val, exp):
    assert parse_float_env(val, 99.0) == exp


@pytest.mark.parametrize("val,exp", [
    ("30", 30),
    ("30.0   # secs", 30),
    (" 30 ", 30),
    ("", 7),
    (None, 7),
    ("nope", 7),
    (30, 30),
])
def test_parse_int_env(val, exp):
    assert parse_int_env(val, 7) == exp


@pytest.mark.parametrize("val,exp", [
    ("true", True), ("false", False),
    ("true   # enable", True),
    ("1", True), ("0", False),
    ("yes", True), ("off", False),
    ("", True),          # blank -> default(True)
    (None, True),
    ("garbage", True),   # invalid -> default(True)
    (True, True), (False, False),
])
def test_parse_bool_env(val, exp):
    assert parse_bool_env(val, True) == exp


def test_orchestrator_filter_signal_survives_commented_cooldown(monkeypatch):
    """The exact reported crash: UNDERLYING_SIGNAL_COOLDOWN with inline comment."""
    monkeypatch.setenv("UNDERLYING_SIGNAL_COOLDOWN", "30.0     # Seconds between signals on same underlying")
    from nifty_scalper_bot.config.env_utils import parse_float_env
    import os
    # Must not raise and must yield 30.0
    assert parse_float_env(os.getenv("UNDERLYING_SIGNAL_COOLDOWN"), 10.0) == 30.0
