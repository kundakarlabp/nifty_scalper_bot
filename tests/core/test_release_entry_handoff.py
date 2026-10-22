"""Regression contracts for atomic broker-flat deployment entry handoff."""

from __future__ import annotations

import fcntl
import os

import pytest

from nifty_scalper_bot.core.trading_switch import (
    TradingSwitch,
    live_entry_submission_slot,
    release_entry_hold_active,
)


@pytest.fixture
def release_config(tmp_path, monkeypatch):
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    return tmp_path


def test_release_hold_blocks_armed_switch_and_live_buy(release_config) -> None:
    switch = TradingSwitch()
    assert switch.arm_for_runtime() is True
    assert switch.can_trade() is True
    marker = release_config / "auto_release_entry_hold"
    marker.touch()
    assert release_entry_hold_active() is True
    assert switch.can_trade() is False
    assert switch.snapshot().can_trade is False
    assert switch.arm_for_runtime() is False
    with pytest.raises(RuntimeError, match="release_entry_hold"):
        with live_entry_submission_slot({"side": "BUY"}):
            pytest.fail("BUY may not reach the broker during release")
    marker.unlink()
    assert switch.can_trade() is True


def test_live_entry_keeps_shared_lock_through_broker_call(release_config) -> None:
    lock = release_config / "auto_release_entry_lock"
    with live_entry_submission_slot({"side": "BUY"}):
        contender = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    contender = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(contender, fcntl.LOCK_UN)
        os.close(contender)


def test_protective_sell_bypasses_entry_barrier(release_config) -> None:
    lock = release_config / "auto_release_entry_lock"
    lock.touch()
    marker = release_config / "auto_release_entry_hold"
    marker.touch()
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exits must remain available while the release is awaiting broker-flat.
        with live_entry_submission_slot({"side": "SELL"}):
            pass
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_shadow_mode_is_not_affected_by_host_release_hold(
    release_config, monkeypatch
) -> None:
    (release_config / "auto_release_entry_hold").touch()
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    with live_entry_submission_slot({"side": "BUY"}):
        pass


def test_live_entry_fails_closed_if_lock_directory_unavailable(
    release_config, monkeypatch
) -> None:
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(release_config / "missing"))
    with pytest.raises(OSError):
        with live_entry_submission_slot({"side": "BUY"}):
            pytest.fail("unknown lock authority cannot permit a live entry")
