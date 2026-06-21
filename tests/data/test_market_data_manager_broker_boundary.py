import asyncio
import math

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.utils.errors import BrokerAuthenticationError, BrokerBalanceUnavailableError


class _WS:
    on_tick = None

    def subscribe_tokens(self, *_args, **_kwargs):
        return None

    def is_connected(self):
        return True


class _SyncBroker:
    def __init__(self, value=0.0):
        self.value = value
        self.calls = 0

    def get_available_balance(self, segment="equity"):
        self.calls += 1
        return self.value


class _AsyncBroker:
    def __init__(self, value=0.0):
        self.value = value
        self.calls = 0

    async def get_available_balance(self, segment="equity"):
        self.calls += 1
        return self.value


@pytest.mark.asyncio
async def test_async_account_snapshot_accepts_async_broker_zero():
    mdm = MarketDataManager(_AsyncBroker(0.0), _WS())
    snapshot = await mdm.get_account_snapshot(force=True)
    assert snapshot["available"] == 0.0
    assert snapshot["valid"] is True
    assert snapshot["stale"] is False


@pytest.mark.asyncio
async def test_async_account_snapshot_runs_sync_broker():
    broker = _SyncBroker(123.45)
    mdm = MarketDataManager(broker, _WS())
    assert await mdm.refresh_available_balance(force=True) == pytest.approx(123.45)
    assert broker.calls == 1


def test_sync_balance_rejects_async_broker_without_leaking_coroutine():
    mdm = MarketDataManager(_AsyncBroker(10.0), _WS())
    with pytest.raises(BrokerBalanceUnavailableError, match="async_broker_method_used_in_sync_context"):
        mdm.get_available_balance(force=True)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, -math.inf])
def test_sync_balance_rejects_invalid_values(value):
    mdm = MarketDataManager(_SyncBroker(value), _WS())
    with pytest.raises(BrokerBalanceUnavailableError):
        mdm.get_available_balance(force=True)


def test_sync_balance_uses_fresh_cache_without_broker_call():
    broker = _SyncBroker(5.0)
    mdm = MarketDataManager(broker, _WS())
    assert mdm.get_available_balance(force=True) == 5.0
    broker.value = 99.0
    assert mdm.get_available_balance(force=False) == 5.0
    assert broker.calls == 1


@pytest.mark.asyncio
async def test_expired_cache_not_returned_after_failed_refresh():
    class Broker(_SyncBroker):
        def get_available_balance(self, segment="equity"):
            self.calls += 1
            if self.calls == 1:
                return 10.0
            raise RuntimeError("network")

    broker = Broker()
    mdm = MarketDataManager(broker, _WS())
    mdm._account_cache_ttl = 0.0
    assert await mdm.refresh_available_balance(force=True) == 10.0
    with pytest.raises(BrokerBalanceUnavailableError):
        await mdm.get_account_snapshot(force=True)
    assert mdm._account_snapshot == {}


@pytest.mark.asyncio
async def test_auth_failure_clears_cache_and_propagates():
    class Broker(_SyncBroker):
        def get_available_balance(self, segment="equity"):
            raise BrokerAuthenticationError("bad token")

    mdm = MarketDataManager(Broker(), _WS())
    mdm._account_snapshot = {"available": 10.0}
    mdm._account_updated_at = 1.0
    with pytest.raises(BrokerAuthenticationError):
        await mdm.get_account_snapshot(force=True)
    assert mdm._account_snapshot == {}


def test_sync_cached_balance_rejected_when_broker_auth_invalid():
    broker = _SyncBroker(10.0)
    mdm = MarketDataManager(broker, _WS())
    assert mdm.get_available_balance(force=True) == 10.0
    broker.auth_invalid = True
    with pytest.raises(BrokerAuthenticationError):
        mdm.get_available_balance(force=False)
    assert mdm._account_snapshot == {}


@pytest.mark.asyncio
async def test_async_cached_balance_rejected_when_broker_auth_invalid():
    broker = _AsyncBroker(10.0)
    mdm = MarketDataManager(broker, _WS())
    assert await mdm.refresh_available_balance(force=True) == 10.0
    broker.auth_invalid = True
    with pytest.raises(BrokerAuthenticationError):
        await mdm.get_account_snapshot(force=False)
    assert mdm._account_snapshot == {}


def test_sync_generic_refresh_failure_clears_cache_and_raises_typed():
    class Broker(_SyncBroker):
        def get_available_balance(self, segment="equity"):
            self.calls += 1
            if self.calls == 1:
                return 10.0
            raise RuntimeError("boom")

    broker = Broker()
    mdm = MarketDataManager(broker, _WS())
    assert mdm.get_available_balance(force=True) == 10.0
    with pytest.raises(BrokerBalanceUnavailableError):
        mdm.get_available_balance(force=True)
    assert mdm._account_snapshot == {}
    with pytest.raises(BrokerBalanceUnavailableError):
        mdm.get_available_balance(force=False)
