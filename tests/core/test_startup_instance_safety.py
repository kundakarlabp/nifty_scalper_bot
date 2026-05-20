import pytest

from nifty_scalper_bot.core.app import _enforce_live_single_replica_safety


def test_live_multi_replica_fail_fast(monkeypatch):
    monkeypatch.setenv('RAILWAY_REPLICA_COUNT', '2')
    monkeypatch.setenv('BOT_FAIL_FAST_ON_MULTI_REPLICA_LIVE', 'true')
    with pytest.raises(RuntimeError, match='Multiple live trading replicas are unsafe'):
        _enforce_live_single_replica_safety(is_live_execution=True)


def test_live_multi_replica_warn_only(monkeypatch):
    monkeypatch.setenv('RAILWAY_REPLICA_COUNT', '2')
    monkeypatch.setenv('BOT_FAIL_FAST_ON_MULTI_REPLICA_LIVE', 'false')
    _enforce_live_single_replica_safety(is_live_execution=True)
