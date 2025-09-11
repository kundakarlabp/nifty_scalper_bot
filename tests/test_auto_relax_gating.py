from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        pass


def test_min_score_relaxes_after_inactivity(monkeypatch) -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.strategy_cfg = SimpleNamespace(min_score=0.35)
    runner.strategy.auto_relax_enabled = True
    runner.strategy.auto_relax_after_min = 30
    now = datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner._last_trade_time = now - timedelta(minutes=31)
    assert runner._min_score_threshold() == 0.30
    assert runner._auto_relax_active is True
    runner._last_trade_time = now - timedelta(minutes=61)
    assert runner._min_score_threshold() == 0.25
    runner._last_trade_time = now - timedelta(minutes=5)
    assert runner._min_score_threshold() == 0.35
    assert runner._auto_relax_active is False


def test_state_shows_auto_relax_banner(monkeypatch) -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.strategy_cfg = SimpleNamespace(min_score=0.35)
    runner.strategy.auto_relax_enabled = True
    runner.strategy.auto_relax_after_min = 30
    now = datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner._last_trade_time = now - timedelta(minutes=31)
    runner._min_score_threshold()
    monkeypatch.setattr(runner, "_within_trading_window", lambda *a, **k: True)
    stub_exec = SimpleNamespace(
        open_count=0,
        get_positions_kite=lambda: {},
        api_health=lambda: {},
        router_health=lambda: {},
    )
    runner.executor = stub_exec
    runner.order_executor = stub_exec
    runner.data_source = SimpleNamespace(health=lambda: {"status": "ok"}, api_health=lambda: {})
    monkeypatch.setattr(runner, "_portfolio_delta_units", lambda: 0)
    snap = runner.get_status_snapshot()
    assert "auto_relax" in snap.get("banners", [])
    assert snap["minutes_since_last_trade"] >= 31


def _prep_settings(monkeypatch) -> None:
    from src.config import settings
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )


def test_why_shows_auto_relax_banner(monkeypatch) -> None:
    from src.notifications.telegram_controller import TelegramController

    _prep_settings(monkeypatch)
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.strategy_cfg = SimpleNamespace(min_score=0.35)
    runner.strategy.auto_relax_enabled = True
    runner.strategy.auto_relax_after_min = 30
    now = datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner._last_trade_time = now - timedelta(minutes=31)
    runner._min_score_threshold()
    monkeypatch.setattr(runner, "_within_trading_window", lambda *a, **k: True)
    stub_exec = SimpleNamespace(
        open_count=0,
        get_positions_kite=lambda: {},
        api_health=lambda: {},
        router_health=lambda: {},
    )
    runner.executor = stub_exec
    runner.order_executor = stub_exec
    runner.data_source = SimpleNamespace(health=lambda: {"status": "ok"}, api_health=lambda: {})
    monkeypatch.setattr(runner, "_portfolio_delta_units", lambda: 0)
    status = runner.get_status_snapshot()
    tc = TelegramController(status_provider=lambda: status, last_signal_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/why"}})
    assert any("auto_relax" in m for m in sent)
