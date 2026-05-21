import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager


class DummyBroker:
    pass


class DummyPositionManager:
    pass


class DummyRateLimiter:
    pass


def test_execution_mode_helpers_belong_to_order_manager_not_guardpair():
    assert hasattr(OrderManager, "_env_truthy")
    assert hasattr(OrderManager, "_execution_mode_env")
    assert hasattr(OrderManager, "_live_flag_enabled")
    assert hasattr(OrderManager, "_order_live_execution_enabled")


def test_execution_mode_env_does_not_raise_attribute_error(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    manager = OrderManager(
        broker_client=DummyBroker(),
        position_manager=DummyPositionManager(),
        rate_limiter=DummyRateLimiter(),
    )

    assert manager.execution_mode == "SHADOW"
    assert manager.get_execution_mode() == "SHADOW"
    assert manager.is_live_mode() is False


def test_live_mode_requires_live_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="LIVE mode requires"):
        OrderManager(
            broker_client=DummyBroker(),
            position_manager=DummyPositionManager(),
            rate_limiter=DummyRateLimiter(),
        )
