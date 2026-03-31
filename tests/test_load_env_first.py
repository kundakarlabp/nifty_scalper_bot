from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_load_env_first_calls_load_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_load_dotenv(*, override: bool) -> None:  # type: ignore[override]
        calls["override"] = override

    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=_fake_load_dotenv)
    )
    sys.modules.pop("nifty_scalper_bot.load_env_first", None)

    module = importlib.import_module("nifty_scalper_bot.load_env_first")
    importlib.reload(module)

    assert calls.get("override") is False
