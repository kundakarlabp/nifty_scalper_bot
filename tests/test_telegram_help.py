from __future__ import annotations

from types import SimpleNamespace

from pathlib import Path
import re

from src.config import settings
from src.notifications.telegram_controller import TelegramController


def _prep_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )


EXPECTED_COMMANDS = {
    "/active",
    "/apihealth",
    "/atm",
    "/atrmin",
    "/atrp",
    "/audit",
    "/backtest",
    "/bars",
    "/cancel",
    "/cancel_all",
    "/cb",
    "/check",
    "/components",
    "/conf",
    "/config",
    "/depthmin",
    "/diag",
    "/diagstatus",
    "/diagtrace",
    "/emergency_stop",
    "/eventguard",
    "/events",
    "/expiry",
    "/filecheck",
    "/force_eval",
    "/fresh",
    "/greeks",
    "/hb",
    "/health",
    "/healthjson",
    "/help",
    "/l1",
    "/lastplan",
    "/lasttrades",
    "/limits",
    "/logs",
    "/logtail",
    "/micro",
    "/microcap",
    "/micromode",
    "/minscore",
    "/mode",
    "/nextevent",
    "/orders",
    "/pause",
    "/plan",
    "/positions",
    "/probe",
    "/quotes",
    "/range",
    "/reconcile",
    "/reload",
    "/resume",
    "/risk",
    "/riskresettoday",
    "/router",
    "/score",
    "/selftest",
    "/shadow",
    "/sizer",
    "/slmult",
    "/smoketest",
    "/start",
    "/state",
    "/status",
    "/summary",
    "/tick",
    "/tpmult",
    "/trace",
    "/traceoff",
    "/trend",
    "/warmup",
    "/watch",
    "/why",
}


def test_help_lists_all_commands(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    cmds = tc._list_commands()
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/help"}})
    msg = sent[0]
    for cmd in cmds:
        assert cmd in msg


def test_expected_commands_are_wired(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    tc = TelegramController(status_provider=lambda: {})
    cmds = set(tc._list_commands())
    assert cmds == EXPECTED_COMMANDS


def test_readme_command_block_matches_handler(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    tc = TelegramController(status_provider=lambda: {})
    commands_line = " ".join(tc._list_commands())
    text = Path("README.md").read_text(encoding="utf-8")
    block = re.search(r"```\n(/[^`]+)\n```", text)
    assert block, "Expected command block missing from README.md"
    assert block.group(1) == commands_line
