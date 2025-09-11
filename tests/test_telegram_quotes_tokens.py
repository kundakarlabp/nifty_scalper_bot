from __future__ import annotations

from types import SimpleNamespace

from src.config import settings
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.strategies.runner import StrategyRunner


def _prep_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )


def test_quotes_reports_atm_tokens(monkeypatch) -> None:
    _prep_settings(monkeypatch)

    runner = SimpleNamespace(
        data_source=SimpleNamespace(atm_tokens=(111, 222), current_atm_strike=17050)
    )
    monkeypatch.setattr(StrategyRunner, "_SINGLETON", runner)

    exe = OrderExecutor(kite=None)
    monkeypatch.setattr(
        "src.strategies.scalping_strategy._token_to_symbol_and_lot",
        lambda k, t: (f"SYM{t}", 50),
    )
    def boom(*args, **kwargs):
        raise AssertionError("resolver called")

    monkeypatch.setattr("src.utils.strike_selector.resolve_weekly_atm", boom)

    sent: list[str] = []
    tc = TelegramController(status_provider=lambda: {}, quotes_provider=exe.quote_diagnostics)
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/quotes"}})

    msg = sent[0]
    assert "ATM strike 17050" in msg
    assert "CE tsym=SYM111" in msg
    assert "PE tsym=SYM222" in msg
