import pytest

from nifty_scalper_bot.core.app import (
    _compute_live_execution_enabled,
    _enforce_live_single_replica_safety,
)


def test_live_multi_replica_fail_fast(monkeypatch):
    monkeypatch.setenv('RAILWAY_REPLICA_COUNT', '2')
    monkeypatch.setenv('BOT_FAIL_FAST_ON_MULTI_REPLICA_LIVE', 'true')
    with pytest.raises(RuntimeError, match='Multiple live trading replicas are unsafe'):
        _enforce_live_single_replica_safety(is_live_execution=True)


def test_live_multi_replica_warn_only(monkeypatch):
    monkeypatch.setenv('RAILWAY_REPLICA_COUNT', '2')
    monkeypatch.setenv('BOT_FAIL_FAST_ON_MULTI_REPLICA_LIVE', 'false')
    _enforce_live_single_replica_safety(is_live_execution=True)


def test_enable_live_env_is_treated_as_live_for_replica_safety(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'PAPER')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('RAILWAY_REPLICA_COUNT', '2')
    monkeypatch.setenv('BOT_FAIL_FAST_ON_MULTI_REPLICA_LIVE', 'true')
    assert _compute_live_execution_enabled() is True
    with pytest.raises(RuntimeError, match='Multiple live trading replicas are unsafe'):
        _enforce_live_single_replica_safety(is_live_execution=True)
