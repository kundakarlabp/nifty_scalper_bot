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


class _SubmittingBroker:
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        return {"order_id": "SIM-1", "status": "success"}


class _MarkedSubmittingBroker(_SubmittingBroker):
    is_simulated_adapter = True


def _live_sim_order_manager(tmp_path, broker):
    from nifty_scalper_bot.execution.position_manager import PositionManager
    from nifty_scalper_bot.utils.rate_limiter import RateLimiter

    return OrderManager(
        broker_client=broker,
        position_manager=PositionManager(str(tmp_path / "positions.json")),
        rate_limiter=RateLimiter(),
    )


def test_live_simulation_order_submission_rejects_unmarked_broker(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _SubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)

    with pytest.raises(RuntimeError, match="non-simulated broker"):
        manager._submit_order_with_retry(  # noqa: SLF001 - canonical broker submission guard
            {"symbol": "NFO:NIFTY26AUG25000CE", "side": "BUY", "quantity": 75}
        )

    assert broker.calls == 0


def test_live_simulation_order_submission_allows_marked_broker(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)

    response = manager._submit_order_with_retry(  # noqa: SLF001 - canonical broker submission guard
        {"symbol": "NFO:NIFTY26AUG25000CE", "side": "BUY", "quantity": 75}
    )

    assert response["order_id"] == "SIM-1"
    assert broker.calls == 1
